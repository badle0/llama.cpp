# Upstream precedents for the S1 core–plugin design, and zero copy (2026-10-28)

The user's question (2026-10-28): the S1 design should stay as close as possible to llama.cpp's existing structure,
since this part may go upstream; find the backend core upstream that resembles ours, analyse its structure, decide
whether S1 needs changing; and explain the "zero copy ability" SpacemiT's AI engineer asked the core to have, and
where it would go. Everything below was read in the code; nothing was built or run. Line numbers marked `U` are
upstream master at commit `8ca4060b` (2026-10-28), read in a scratch clone; unmarked ones are this tree (base
`ba360ef`, 2026-08-11). Inferences and readings of documents are labelled as such.

## 0. Summary

- There is no single upstream "backend core" like ours. Three upstream pieces together cover what S1 does, and S1
  matches each of them: ggml's own backend loader (the plugin boundary: a versioned function table behind `dlopen`,
  §3.1), Metal's split of an op layer from a device layer behind a pure C header with opaque handles (§3.2), and the
  host/device split of Hexagon and ET, where the host builds a list of op descriptors and the device replays it with
  a barrier per step (§3.3, §3.4). The one S1 element with no upstream precedent is the inner `dlopen`: ggml's plugin
  unit is the whole backend. So the plugin must also build linked in (A5), the way upstream would take it.
- Decision: S1's shape stands. Seven alterations are proposed before the first S1 spec, five of them small (§4):
  the entry and version check as ggml's loader does them, optional functions through `get_proc_address`, names
  after `ggml-backend-impl.h`, the plan cache keyed by the split's `uid` (which our base already stamps), the entry
  symbol usable with `dlopen` or linked in, two reserved memory entries for zero copy, and the precedents recorded for
  the choices that stay as they are.
- Zero copy, in ggml's vocabulary, is one of two things: the model file used in place (`buffer_from_host_ptr`: the
  CPU backend, Metal, virtgpu) or device buffers in shared pages that the CPU writes directly, with no staging copy
  (Hexagon). It is never `is_host`: upstream defines `is_host` as "host memory and standard ggml tensor layout"
  (`U ggml/src/ggml-backend-impl.h:31`), the loader writes raw file bytes into such buffers (`U src/llama-model-loader.cpp:1749-1751`),
  and every upstream backend with a layout of its own reports it false, Metal and Hexagon included (§3.5, §5.1).
- The provider already has the shared-memory half: it reads host buffers in place and its own buffers are host DRAM
  (`flagos-spacemit.cpp:895`, `:225`). It lacks the mapped file: `caps.buffer_from_host_ptr` is false (`:858`,
  `:907`), so every tensor is copied once at load, repacked or not. Proposed (§5.4): a mapped buffer type as Metal's,
  with the IME copy built at `init_tensor` beside the mapped bytes and kept in `tensor->extra` as Hexagon does, so that
  every tensor the IME does not need repacked is used from the file and `get_tensor` becomes exact for every layout.
  A host-visible KV cache (a plain default type plus a repack extra type through llama.cpp's extra-buffer-type hook)
  is possible but changes placement and test coverage; proposed only if a supported configuration shows copies that
  matter. GGUF-layout IME kernels, the deck's ② in full, belong to M3.

## 1. What was read

- Upstream: `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml-backend-reg.cpp`, `ggml-backend-dl.{h,cpp}`,
  `ggml-backend.cpp`, `ggml-backend-meta.cpp`; `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/llama-context.cpp`,
  `src/llama-kv-cache.cpp`, `src/llama-graph.cpp`; `ggml/src/ggml-metal/*`, `ggml-hexagon/*` and `htp/*`, `ggml-et/*`,
  `ggml-cuda/ggml-cuda.cu`, `ggml-rpc/ggml-rpc.cpp`, `ggml-virtgpu/*`, `ggml-cpu/{traits.h,repack.cpp,ggml-cpu.cpp}`,
  `ggml-cpu/spacemit/*`; `tests/test-backend-ops.cpp`; `docs/backend/*`. Three read-only surveys ran in parallel and
  every line this note rests on was then read again directly.
- Ours: `plan.md` "S1 design" and §2.3, `structural-review.md` §3 and §4.1, `structural-experiments.md` F2 and F4,
  `flagos-spacemit.cpp` (the buffer type and device vtables).
- Upstream has grown since our base: backends `ggml-hexagon`, `ggml-virtgpu`, `ggml-et`, `ggml-openvino`,
  `ggml-zendnn`; `GGML_BACKEND_API_VERSION` 3 (ours 2; `U ggml-backend-impl.h:11`, ours `:11`); a `META` device type
  for tensor parallelism (`U ggml-backend.h:145-146`); `caps.mmap_support` (`:159-160`); `alloc_buffer_n` and
  `get_alloc_size_n` in the buffer-type interface (`U ggml-backend-impl.h:21-22`, `:29-30`); 2-D tensor copies; the
  `graph_optimize` backend hook (`:156`).

## 2. Upstream's boundaries in one table

| Boundary | Mechanism | What crosses | Version check | Zero copy | Files |
|---|---|---|---|---|---|
| ggml ↔ a backend library | `dlopen(RTLD_NOW \| RTLD_LOCAL)`, entry `ggml_backend_init`, optional `ggml_backend_score`; file pattern `[lib]ggml-<name>-*.so` next to the executable, `GGML_BACKEND_PATH` for one out of tree | a `ggml_backend_reg` (function table + opaque `context`); optional functions by `get_proc_address` | `api_version` compared for exact equality, after init; a mismatch logs and unloads | n/a | `U ggml-backend-reg.cpp:221-261`, `:466-498`, `:578-596`; `ggml-backend-dl.cpp:35` |
| Metal: op layer ↔ device layer | one process, one library; a C header with opaque handles (device, library, pipeline, encoder, buffer, event) implemented in Objective-C | per-op encoder calls (set pipeline, bytes, buffers, dispatch); pipelines looked up by name with a cache | none (one build) | mapped model file (`newBufferWithBytesNoCopy`); `is_host` false on all three buffer types | `U ggml-metal/ggml-metal-device.h`, `-ops.cpp:4601-4625`, `-device.cpp:2264-2284`, `ggml-metal.cpp:288-292, :366-370, :444-448, :704-710, :832` |
| Hexagon: host ↔ DSP | FastRPC session per device; the DSP library `libggml-htp-v<arch>.so` opened by URI; the host dlopens `libcdsprpc` | batches of up to 2048 op descriptors (opcode, op params, a 128-byte kernel-parameter blob solved on the host, tensor and buffer indices) in a pinned shared block; up to 8 batches in flight; completion once per batch; fences in the stream | none found (struct-size checks only; inferred: a mismatched build with equal sizes would pass) | buffers are `rpcmem` pages mapped into the DSP by fd; the CPU `memcpy`s into them directly; no `buffer_from_host_ptr`, `mmap_support` false; `is_host` false | `U ggml-hexagon/htp/htp-ops.h:149-181, :257-279`; `ggml-hexagon.cpp:147-148, :1764-1847, :2364-2434, :669-690, :968-976, :2952-2969, :3034` |
| ET: host ↔ ET-SOC "uberkernel" | device ELFs embedded and uploaded by the vendor runtime | an instruction list `{kernel_id, flags, params_offset, params_size}` plus a parameter blob, one launch per graph segment, a global barrier between steps | none | n/a | `U ggml-et/ggml-et-uberkernel-common.h:5-17`; `et-kernels/src/uberkernel.c:239-247`; `ggml-et-kernels.cpp:339-353, :406-430` |
| CUDA graphs | in-process capture and replay | the split's graph captured in chunks of 512 nodes; replayed per token; kernel parameters updated with `cudaGraphExecUpdate` | n/a | n/a | `U ggml-cuda/common.cuh:1264-1292`; `ggml-cuda.cu:2797-2869, :2872-2878, :4922-4945` |
| RPC, virtgpu | sockets; virtio shared memory | the whole ggml graph (node ids and a flat tensor array), rebuilt on the other side and run by the stock backend's `graph_compute`; recompute by `uid` | RPC 8.0.1, major equal and minor not newer; APIR 0.2 handshake | virtgpu: the host wraps the guest's blob with `buffer_from_host_ptr` when the device has it | `U ggml-rpc/ggml-rpc.cpp:1223, :1249-1255, :2194`; `ggml-virtgpu/backend/backend.cpp:65-91`, `backend-dispatched-device.cpp:126-139` |
| CPU extra buffer types (AMX, KleidiAI, repack, SpacemiT IME) | compiled in; per-tensor traits in `tensor->extra` | kernels chosen per tensor by the buffer type's traits | n/a | `is_host` null (false); repack in `set_tensor`; `get_tensor` null | `U ggml-cpu/traits.h:20-32`; `ggml-cpu.cpp:42-74, :490`; `repack.cpp:5253`; `spacemit/ime.cpp:1678` |

## 3. The precedents

### 3.1 ggml's own plugin boundary: the backend loader

What ggml loads is a whole backend. `ggml_backend_load` opens the library with `RTLD_NOW | RTLD_LOCAL`
(`U ggml-backend-dl.cpp:35`), looks up `ggml_backend_score` and returns nothing when it answers 0 (`U ggml-backend-reg.cpp:229-232`),
looks up `ggml_backend_init` (`:237`), calls it, and refuses the result when `reg->api_version != GGML_BACKEND_API_VERSION`
(`:246`, `:252-256`): an exact match of one integer, checked after the backend's init has run. Optional functions are not
members of the table; they are reached by name through `get_proc_address` (`U ggml-backend-impl.h:238`; the typedefs at
`U ggml-backend.h:209-227`), and llama.cpp finds extra buffer types, split buffers and thread settings the same way.
A loaded backend is never unloaded at exit (`:177-181`, a FIXME). The entry takes no host table: a backend links
`ggml-base` and logs through it. `load_all` scans `[lib]ggml-<name>-*.so` in the executable's directory and the current
one, plus `GGML_BACKEND_DIR` (`:466-498`), then `GGML_BACKEND_PATH` for one out-of-tree library (`:593-596`). The provider
loaded this way is what E4 measured, on `ba360ef`; against today's master the module would be refused by the version
check (2 against 3) until the mentor's tree is rebased.

### 3.2 Metal: an op layer over a device layer behind a C header

Metal is the only upstream backend that separates what we call the core from what we call the plugin, and it does it
with the shape S1 chose. `ggml-metal-device.h` is a pure C header of opaque handles, `ggml_metal_device_t`,
`ggml_metal_library_t`, `ggml_metal_pipeline_t`, `ggml_metal_encoder_t`, `ggml_metal_buffer_t`, `ggml_metal_event_t`
(`U ggml-metal/ggml-metal-device.h:9-14, :33, :67-73, :98, :315, :353`), with function families for the device
(`:320-347`), buffers (`:355-372`: `init`, `map` for a host pointer, `free`, `get_base`, `set_tensor`, `get_tensor`,
`cpy_tensor`, `clear`, `get_id`), the library (`:100-108`: a pipeline by name, with a cache, `-device.cpp:2264-2284`)
and the encoder (`:75-92`). The op layer, `ggml-metal-ops.cpp`, is C++ that never sees a Metal type: for `RMS_NORM` it
resolves buffer ids from the tensors (`:22-34`), fills a kernel-argument struct, asks the library for the pipeline by
name (`:4601-4603`) and encodes set-pipeline, set-bytes, set-buffer, dispatch (`:4616-4625`). The device layer is
Objective-C (`-device.m`, `-context.m`). The two are one library; nothing is `dlopen`ed. Upstream's note on why the
split exists is only that the helpers are "too difficult to implement in Objective-C" (`ggml-metal-common.h:1`); the
reorder pass in `-common.cpp` says of itself "generic and not specific to metal" (`:49-50`).

### 3.3 Hexagon: a host core and a DSP runtime, the K3's nearest relative

Hexagon is the upstream backend whose device looks most like the K3's AI cores: a worker pool of up to 10 HVX threads
with one small fast memory (VTCM) that the kernels carve up, driven by a host that cannot run the kernels itself.

- The host is the core. It maps ggml ops to opcodes and decides support and every kernel parameter in one place,
  the "solvers" of `htp-kparams.cpp`: kernel type (HVX or HMX), thread count, chunk sizes, prefetch depth, the VTCM
  layout size (`U ggml-hexagon/htp/matmul-ops.h:61-88` lists the fields). The DSP re-derives nothing
  (`U docs/backend/snapdragon/developer.md:240-246`). It fuses at enqueue time (MUL_MAT+ADD, RMS_NORM+MUL, several
  matmuls sharing an activation, `htp-fusion.cpp:618-649`) and reorders matmuls that share an activation through
  `graph_optimize` (`ggml-hexagon.cpp:2563-2618`). It caches the converted node list by the graph's `uid`
  (`:2376-2423`).
- What crosses is a list, not a graph: up to 2048 op descriptors per batch, each an opcode, the op's params, a
  128-byte kernel-parameter blob and 16-bit indices into a tensor table (offset plus buffer index, type, `ne`, `nb`,
  flags) and a buffer table (fd, base, size), all in a pinned shared block; only a 24-byte header goes through the
  queue (`htp/htp-ops.h:149-181, :257-266`; `ggml-hexagon.cpp:1596-1611`). Up to 8 batches are in flight, completion
  comes once per batch, errors keep the batch running and surface at the next `graph_compute` or `synchronize`
  (`:1711-1750`, `:2367-2369`); events are fence ops inside the stream (`:2801-2835`).
- The DSP is a flat interpreter: a switch per opcode (`htp/main.c:898-1052`), `work_queue_run(func, data, n)` calling
  `func(n, i, data)` on every thread with each kernel partitioning its own work (`htp/work-queue.c:123-124`), the whole
  VTCM acquired once per session as a single page (`main.c:328-363`) and carved per op with layout macros into
  per-thread regions (`htp/htp-vtcm.h:13-17`; `matmul-ops.c:273`), the host's size checked again on the DSP
  (`matmul-ops.c:1874-1878`). A quantized matmul DMAs the activations into VTCM, quantizes them with all threads into a
  shared VTCM buffer, then each thread streams 32-row weight tiles into its own VTCM ring with a prefetch depth of 2 to
  16 and writes F32 results to DDR (`matmul-ops.c:1717-1954`, `:251-348`). This is our tile loop, our agreed TCM and
  our path B, in another vendor's words.
- Buffers: the default type is `rpcmem` pages mapped lazily into the DSP by fd (`:442-453`, `:1938-1948`); the CPU reads
  and writes them directly and the DSP maps the same pages, so nothing is staged. `set_tensor` repacks into 32-row tiles
  only when the buffer's usage is `WEIGHTS` and the tensor qualifies (`:669-690`); `get_tensor` un-repacks (`:712-730`);
  `get_alloc_size` grows the allocation for the tiles (`:929-939`); a weight that was not tiled at load is repacked in
  place at its first `graph_compute` (`:2394-2407`, `GGML_HEXAGON_RT_REPACK`), which is how test-backend-ops' weights get
  tiled. `is_host` is false (`:968-971`); a second, host buffer type exists for debugging only (`:973-976`,
  `:2977-2983`). The device reports GPU; `async` and `events` true, `host_buffer`, `buffer_from_host_ptr` and
  `mmap_support` false (`:2952-2969`); no extra buffer types.
- Diagnostics by environment: `GGML_HEXAGON_OPFILTER` (a regex of ops to refuse, the deck's `DISABLE_OPS`),
  `OPFUSION` (a bitmask), `OPBATCH`, `OPQUEUE`, `PROFILE` (modes 0-3 with PMU counters), `OPTRACE` with a Perfetto
  converter, `VERBOSE` (`:84-152`, `:3391-3416`; `U scripts/snapdragon/`).

### 3.4 What crosses: ET's uberkernel, Hexagon's batches, CUDA's uid-keyed replay, RPC's whole graph

The question of 2026-10-27, whether the core sends a graph or a kernel, has three upstream answers. RPC and virtgpu
send the whole ggml graph and let the stock backend's `graph_compute` run it (`U ggml-rpc.cpp:1223, :2194`). CUDA
captures the split into a CUDA graph and replays it per token: the cache is keyed by the split's first node
(`U ggml-cuda.cu:2797-2799`), the fast path returns when the split's `uid` is unchanged (`:2807-2814`), otherwise node
properties are compared (`:2819-2836`) and kernel parameters updated with `cudaGraphExecUpdate` (`:2845-2869`); graphs
are used only after two eager runs without a change (`:4922-4945`). Hexagon and ET send lists: Hexagon's op batches
(§3.3) and ET's opt-in uberkernel, an instruction list `{kernel_id, flags, params_offset, params_size}` with a parameter
blob, one launch per graph segment and `et_barrier_global` between steps (`U ggml-et-uberkernel-common.h:5-17`;
`et-kernels/src/uberkernel.c:239-247`; `ggml-et-kernels.cpp:339-353`). Our step list per split, replayed as one launch
with a barrier per step, is ET's uberkernel with Hexagon's host-solved parameters; CUDA's capture is the same idea
inside one process. The `graph_plan_*` slots of `ggml_backend_i` exist for a plan per graph but are "not used
currently" (`U ggml-backend-impl.h:137-145`), so CUDA keeps its cache inside `graph_compute`, as we would.

### 3.5 Buffers with a layout of their own

Who reports what (read in the code):

| Backend | `is_host` | `buffer_from_host_ptr` | `host_buffer` | `mmap_support` | own layout |
|---|---|---|---|---|---|
| CPU | true | true | n/a | true | no |
| CPU extra types (AMX, KleidiAI, repack, SpacemiT IME) | null, i.e. false (`U repack.cpp:5253`, `ime.cpp:1678`) | n/a | n/a | n/a | repack in `set_tensor`, `get_tensor` null |
| Metal (shared, private, mapped types) | false, all three (`U ggml-metal.cpp:288-292, :366-370, :444-448`) | true (`:832`) | false | true | no |
| Hexagon | false (`:968-971`) | false (`:3034`) | false | false | tiles in `set_tensor` |
| CANN | false (`U ggml-cann.cpp:1585-1588`) | | | | NZ and Q4_0/Q8_0 conversions (F10) |
| OpenCL, RPC, virtgpu | null | | | | |
| ours | false (`flagos-spacemit.cpp:225`) | false (`:858`, `:907`) | false (`:857`, `:906`) | true | IME layouts in `set_tensor`, converted back in `get_tensor` |

The reason is the definition: `is_host` means "tensor data is in host memory and uses standard ggml tensor layout"
(`U ggml-backend-impl.h:31`), and llama.cpp relies on both halves: without mmap it reads the file straight into such a
buffer, bypassing `set_tensor` (`U llama-model-loader.cpp:1749-1751`); the KQ mask and the cache's row indices are
written through `tensor->data` under an `is_host` assert (`U llama-graph.cpp:524`; `llama-kv-cache.cpp:1470-1471`); the
CPU backend accepts a buffer type for its ops when it is host or one of its own extra types (`U ggml-cpu.cpp:490`);
`ggml_backend_tensor_copy` short-circuits through `set_tensor` and `get_tensor` when either side is host
(`U ggml-backend.cpp:635-638`). A buffer that repacks cannot say it, whatever memory it is in; D4's `is_host = false`
is the upstream rule, not a K3 exception.

## 4. S1 against them: the decision and the alterations

S1's shape stands. Element by element:

| S1 element (`plan.md` "S1 design") | Upstream analog | Verdict |
|---|---|---|
| a pure C function table with opaque handles, exported by one entry symbol, loaded with `dlopen` | `ggml_backend_reg` + `ggml_backend_init` (§3.1); Metal's `-device.h` (§3.2); virtgpu's host loader, the only upstream code that `dlopen`s a backend and resolves `ggml_backend_init` itself (`U ggml-virtgpu/backend/backend.cpp:65-91`) | keep; align the entry, the version check and the extension idiom (A1, A2, A3); add the linked-in build (A5) |
| the core keeps the ggml vtables, the contract table, op-to-steps, the plan cache, the loader, diagnostics | Metal's `ggml-metal.cpp` + `-ops.cpp` + `-common.cpp`; Hexagon's host (`htp-kparams`, `htp-fusion`, `htp-opnode`) | keep |
| the plugin keeps device, buffers with the repack inside, stream, kernels, the capture replay | Metal's `-device.m` + `-context.m` + kernels; Hexagon's `htp/` | keep |
| `kernels()` returning the plugin's table of named entries with `supports` and `workspace_size` | Metal's pipeline by name with a cache; Hexagon's solvers deciding support | keep (binding every name at load gives the loader's negative tests; Metal binds lazily) |
| a step list per split, `replay` as one launch with a barrier per step | ET's uberkernel; Hexagon's batches; CUDA's capture (§3.4) | keep |
| the plan cache keyed by the split's structure, a hash then a full compare per graph | CUDA: the split's `uid` first, properties only when it changes; RPC recomputes by `uid` | alter (A4) |
| steps carry `ggml_tensor *`, kernels read the node at run time | Hexagon, ET and Metal pass descriptors or argument structs, because their device is another processor | keep for S1, recorded (A7) |
| the TCM-dependent path choice inside the kernels | Hexagon solves it on the host; the deck's L2 | keep for S1 as decided, precedent recorded (A7) |
| per-tile services: tile index and count, the agreed TCM, the workspace, pair waits, the failure flag | Hexagon's `work_queue_run`, one VTCM carved per op into per-thread regions, the host's size re-checked on the DSP | keep |
| synchronous replay, `caps.async` false | Hexagon, Metal and CUDA are asynchronous with events | keep for S1; `stream_synchronize` is already in the table for a later asynchronous replay |
| failures as statuses, nothing written after a failed check | Hexagon keeps the first error and reports it at the next call; CUDA aborts | keep (ours is stricter and synchronous) |
| diagnostics under the deck's names (`GGML_FLAGOS_*`) | upstream's `GGML_<BACKEND>_*` convention; `GGML_HEXAGON_OPFILTER` is `DISABLE_OPS`, `OPTRACE` is `TRACE_LAUNCH` with a converter | keep |
| memory: `buffer_alloc/free/base`, `tensor_set/get/fill`, the repack inside | Metal's buffer family (`init`, `map`, `set`, `get`, `cpy`); Hexagon's `set_tensor` repack and `extra` | add two entries for zero copy (A6, §5.4) |

The alterations, proposed for the user's review before the first S1 spec (the header and the loader):

- **A1. The entry and the version check as ggml's loader does them.** `const struct spacemit_plugin_i * flagos_spacemit_plugin_init(void)`,
  no host table: the plugin links `ggml-base` and logs through `GGML_LOG_*` like every upstream backend, and the
  plugin directory arrives with M3's package, the way `GGML_BACKEND_PATH` does upstream. The table begins with
  `int api_version`, compared for exact equality (`U ggml-backend-reg.cpp:246`); the `size` field goes, since
  extension happens through A2. The negative tests stay: no library, a wrong version, a missing entry, a missing
  step name, zero devices.
  **Decided (user, 2026-10-28): not adopted; the S1 note's §2 and §6 stand.** The deck's overall structure is kept and
  followed (the entry `flagos_spacemit_plugin_v1(host)` with the host table, the version in the symbol), and `size`
  stays: it is the mentor's own convention (`flagos_provider_v1 { api_version, struct_size }`,
  `ggml/src/ggml-flagos/flagos-provider.h:65-67`, validated as exact version plus `struct_size >= required`,
  `flagos-provider.cpp:5-8`, with one optional tail already in use, `:81-85`), not an addition of this note as §4 first
  said. Upstream is followed where it costs nothing and conflicts with neither the deck nor the mentor: the exact
  version equality and the `dlopen` flags, both already in the note. The host table's `log` is implemented by the core
  as a forward to ggml's logger, so a plugin message reaches `ggml_log_set`'s callback either way; the mentor's AMD
  code itself has no host table (its entry `flagos_amd_provider()` takes no argument and returns a static table,
  `providers/amd/flagos-amd.cpp:6184`), so the host table is the deck's, kept by the user's choice.
- **A2. Optional functions through `get_proc_address(const char * name)`.** The table holds the required functions
  only; the test counters, the host executor's switches and later M3's `module_load` and `kernel_get` are resolved by
  name, as `ggml_backend_reg_i.get_proc_address` does it (`U ggml-backend-impl.h:238`) and as llama.cpp finds
  `ggml_backend_dev_get_extra_bufts`. The loader's check "every required pointer non-null" then has no exceptions,
  and `test_control` leaves the table.
- **A3. Names after `ggml-backend-impl.h`.** The table is `spacemit_plugin_i`; the handles keep their typed opaque
  structs (better than a `void * context`); the step and kernel structs keep their names. Cosmetic, for the reader
  who knows upstream.
- **A4. The plan cache keyed by the split's `uid` first.** The scheduler stamps every split's graph
  (`ggml-backend.cpp:1538` in our base; `U :1751`), and llama.cpp keeps the split when it reuses the graph
  (`src/llama-context.cpp:1347`: no reset or re-allocation on reuse), so in generation a split keeps its `uid` from
  token to token and gets a new one when the graph is rebuilt. CUDA's fast path returns on an equal `uid`
  (`U ggml-cuda.cu:2807-2814`) and compares node properties only then; RPC recomputes by `uid`. The S1 note's §3 (ii),
  a hash and a full compare every graph, becomes: `uid` lookup first; on a new `uid`, the structural compare against
  the cached entries, and a rebuild on a miss. E3c's 0.55 ms per token (the host work the cache was to remove) is then
  bounded by the lookup, and the "kept only if measured" rule stays.
- **A5. The entry symbol usable with `dlopen` or linked in.** Upstream's plugin unit is the backend; an inner
  `dlopen` is the deck's requirement (D10), not an upstream pattern, and a reviewer there would ask why the plugin is
  not simply part of the backend, as Metal's device layer is. So `provider.cmake` gets a switch
  (`FLAGOS_SPACEMIT_STATIC_PLUGIN`): the same header, the same table, the loader calling the entry directly when the
  plugin is linked in and `dlopen`ing it otherwise. The Mac's host plugin and the K3's real one both build either way.
  Cost: a few lines in the loader; benefit: the upstream-shaped form exists from day one.
- **A6. Two memory entries reserved for zero copy** (§5.4): `buffer_wrap(dev, ptr, size, &buf)`, the deck's
  `register_host_ptr`, behind `ggml_backend_device_i.buffer_from_host_ptr`; and `tensor_prepare(t)`, the repack from a
  tensor's own bytes into plugin memory recorded in `t->extra`, behind `init_tensor`. In S1 they are declared and the
  core's `buffer_from_host_ptr` stays null; S2's memory part implements them with the mapped buffer type.
- **A7. Precedents recorded, nothing changed.** (i) The step's `ggml_tensor *` stays for S1: Hexagon's 128-byte
  kernel-parameter blob and ET's parameter blob are the shape it takes when M3's typed launch arrives. (ii) The path
  choice stays in the kernels for S1 as decided; Hexagon's solvers show the later move into the core has an upstream
  form. (iii) `ggml_backend_i.graph_optimize` (`U ggml-backend-impl.h:156`) is where Metal and Hexagon reorder nodes
  before scheduling; it is the hook for the dependency-placed barriers of `fusion-and-splitting.md` when M4 comes.
  (iv) Upstream records an op-coverage CSV per backend under `docs/ops/` (Hexagon's lists 25 ops fully and 45 partly
  supported); an upstream submission would carry one for FlagOS.

What this does not change: the deck's five stages map as `structural-review.md` §4 found; the capture replay stays
the K3's main path (E3a, E3b, F16); D4's storage model stands as the plugin's choice, now with the upstream rule behind
its `is_host` (§3.5).

## 5. Zero copy

### 5.1 In ggml's terms

Three distinct things are called zero copy around ggml; only the first two exist upstream.

1. **The model file used in place.** A device whose `caps.buffer_from_host_ptr` is true gets, for every tensor
   context placed in its default buffer type under mmap, a buffer made from the mapping itself
   (`U src/llama-model.cpp:1940-1948`: the condition is mmap or lazy loading, the cap, and the default type;
   `:1962` the call, one buffer per mapping range). The tensors are then pointed at the file's bytes with
   `ggml_backend_tensor_alloc`, which calls only the buffer's `init_tensor`, never `set_tensor`
   (`U llama-model-loader.cpp:1731-1732`; `U ggml-backend.cpp:2258-2260`). The mapping is read-only
   (`U src/llama-mmap.cpp:495`). Metal does this with `newBufferWithBytesNoCopy` (`U ggml-metal-device.m:2226`), the
   CPU backend by construction (`U ggml-cpu.cpp:398`), virtgpu's host for the guest's blobs. Every tensor so mapped
   costs no load copy and no second resident copy; tensors that need another layout cannot be mapped, since the
   mapping cannot be written.
2. **Shared pages with no staging copy.** On a unified-memory device, the device's buffers are pages the CPU can
   address: `set_tensor` and `get_tensor` are plain `memcpy`s or nothing, the scheduler's copies between the CPU
   backend's buffers and the device's are the only ones left, and they occur only where one side cannot read the other
   (`U ggml-backend.cpp:1523-1543`: a copy is made when the split's backend does not support the source's buffer type).
   Hexagon's `rpcmem` pages mapped into the DSP are this (§3.3).
3. **Host visibility**, `is_host` true on the device's buffers, so the CPU backend may run ops on them in place. No
   upstream unified-memory backend does this (Metal and Hexagon report false, §3.5), and the definition forbids it for
   any buffer with a layout of its own.

A fourth copy is llama.cpp's own and no backend's to remove: a tied output head is loaded twice, once for the input
layer on the CPU and once for the output layer on the device (`TENSOR_DUPLICATED`, `U src/llama-model.cpp:3864`), and
the logits leave any buffer through `ggml_backend_tensor_get_async` (`U src/llama-context.cpp:2081`), the CPU's
included.

### 5.2 What the provider has and lacks

- Has, by the code: the second kind. Its buffers are host DRAM on huge pages (`flagos-spacemit.cpp:102-122`); the
  AI-core threads read any process memory; `set_tensor` is a `memcpy` or the repack and `get_tensor` the reverse; it
  accepts every host buffer type for its operands (`:895`), so the CPU's compute buffers, the graph inputs (which the
  scheduler places on the CPU backend, `U ggml-backend.cpp:1093-1094`; the mask written in place under the host assert)
  and, under ACCEL, the KV cache were all read in place. E2 measured 2 splits per token under IGPU with no copies in
  the default configuration (`structural-experiments.md` §6).
- Lacks: the first kind. `caps.buffer_from_host_ptr` is false and the device has no `buffer_from_host_ptr` (`:858`,
  `:907`), so every tensor is copied once at load through `set_tensor`, the plain ones (norm weights, biases, RoPE
  factors) as well as the repacked ones, and `get_tensor` returns the converted weights for the lossy layouts (D4,
  question 2).
- Copies that remain, by configuration: in the default one (F16 cache, `-fa on`) only the load copy and the logits;
  with a quantized cache, the scheduler copies the cache views for the attention and the shift's Hadamard matmuls that
  stay on the CPU (D3; the Q8_0 run 6.95 against 11.08 t/s on Qwen3-4B, `build.md` §7, the copies' share of that
  not isolated); a state save or load goes through `get_tensor` and `set_tensor` as on any device.

### 5.3 The deck's words, the engineer's remark

The deck puts "unified memory zero-copy" in L0, the ggml interface layer, and says at stage ②, model load, "unified
memory is zero-copy, weights need not be copied again"; its Tier-2 zero-copy functions are `buffer_map` and
`register_host_ptr` (`structural-review.md` §3, the layers table, and §4.7's tiers table). `register_host_ptr` is
`buffer_from_host_ptr` (kind 1); `buffer_map` is `buffer_base`, which the table has. Our reading of the engineer's
remark, not confirmed with them: the core should let a plugin expose the file's bytes to the kernels without copying,
and the core must not insert copies between host memory and plugin buffers. The second holds today. The first is kind
1, which the IME layouts block for the matmul weights (G6) and nothing blocks for every other tensor. The question to
put to them: whether they mean the model file in place (kind 1), or the CPU running ops on the device's buffers (kind
3), which no upstream backend offers and which would change placement (§5.4, option B).

### 5.4 Options and where each goes

| Option | Design | Upstream precedent | Gain | Cost | Where |
|---|---|---|---|---|---|
| **A, proposed first** | a third buffer type, *mapped*, created only by `buffer_from_host_ptr`; `caps.buffer_from_host_ptr` true; its `init_tensor` runs the layout rule on the tensor's own bytes and, when the rule says repack, builds the IME copy into plugin memory on huge pages and records it in `tensor->extra`; `get_alloc_size` is `ggml_nbytes` for this type; `set_tensor` is refused (the mapping is read-only; llama.cpp never writes a mapped weight); `get_tensor` reads the mapped bytes; the allocated default type stays as it is (the KV cache, the compute buffers, weights without mmap); `is_host` false on both, as Metal; `supports_buft` accepts both; the kernels take a weight's pointer from one helper (`extra` when set, else `data`) | Metal's mapped buffer type (`U ggml-metal.cpp:444-480`, `:832`); Hexagon's `extra` and `init_tensor` as "add tensor extras" (`U ggml-backend-impl.h:55`); Hexagon's repack at first use (`U ggml-hexagon.cpp:2394-2407`) | every tensor the IME does not need repacked is used from the file; `get_tensor` exact for every layout (question 2's read-back caveat gone under mmap); the deck's ② met for those tensors; M3's GGUF-layout kernels read mapped tensors directly; when a GGUF-layout IME matmul exists, the side copy disappears with no further ggml-side change | about 150-250 lines plus tests (an estimate); resident memory unchanged (the IME copy as today; the mapping's pages are page cache); `--no-mmap` unchanged | core: the mapped type's vtable and `device_i.buffer_from_host_ptr`; plugin: `buffer_wrap` and `tensor_prepare` (A6); kernels: the pointer helper; S2's memory part |
| B | the default type *plain and host-visible* (`is_host` true, `buffer_from_host_ptr` true), the repack in a second type exposed through `get_proc_address("ggml_backend_dev_get_extra_bufts")`; `supports_op` refuses a matmul whose weight sits in the plain type, so `select_weight_buft` picks the repack type for every claimable weight (the probe allocates the weight in each candidate type in turn, `U llama-model-loader.cpp:1066-1083`; the device's extras come after its default, `U llama-model.cpp:1241-1253`) | the mechanism exists for GPU devices but no upstream backend uses it (only `ggml-cpu` exports the hook); the CPU's extra types are its model | the KV cache and compute buffers readable by the CPU in place: no scheduler copies for a quantized cache's attention and shift; plain tensors mapped as in A | placement changes for every weight; test-backend-ops allocates everything in the default type, so a matmul there needs a run-time repack into plugin memory (Hexagon's in-place form is unavailable, the plain type promising standard layout) or a GGUF-layout kernel; the device would be the first upstream user of that hook; the provider's remaining fallbacks run on the X100 cores anyway | only if a supported configuration shows the copies matter (§5.5); not for S1 |
| C | no repack: IME matmuls on GGUF layout, every weight mapped | Metal (no repack at all); the deck's ② literally; FlagGems' kernels read GGUF layouts | the full deck wording; no `get_tensor` conversion; no repack code | kernel work: the 1-row path and path B stage slabs into TCM already, so their copies could permute on the way (an estimate; unmeasured), paths A and C read tiles from DRAM in IME order and would need a staged variant; speed unknown | M3, with D9's kernel work |

Option A is proposed because it is the smallest change that gives the ggml-visible zero copy upstream has, keeps
`is_host` false like every upstream backend with a layout, needs no placement change, and leaves the matmul weights'
one resident copy where D4 put it until option C removes it.

### 5.5 What settles the open points

- Which kind the engineer meant: ask, with §5.1's three kinds in front of them.
- Option A's correctness and cost: the check program loading Qwen3-0.6B's tensors through `buffer_from_host_ptr` on
  the Mac's simulated device with `get_tensor` byte-identical to the file; then the acceptance runner on the K3
  (identical logits, tg128 and pp128 within 1%, resident memory from `/proc/self/status` not above today's).
- Whether option B is worth its placement change: the Q8_0-cache run on Qwen3-4B with `GGML_SCHED_DEBUG` to count the
  copied bytes per token, and the same run with the copies' time isolated (E3c's counters); only if the copies are a
  measurable share of the 6.95 against 11.08 t/s gap does B's gain exist.
- Option C's speed: the 1-row path's slab copy permuting from GGUF layout, timed by `flagos-check-spacemit --bench`
  against today's `memcpy` staging (the first measurement; paths A and C only after it).

## 6. Side findings

- **SpacemiT's in-tree IME code moved since our base** (`diff` of `ggml/src/ggml-cpu/spacemit/` against `U`):
  an IME1 Q8_0 path, `gemm_kernel_i8i8` with a 16-row interleaved repack `q8_0_16x32` behind `use_ime1`
  (`U ime.cpp:323-325, :629-631, :1251, :1361-1366`; `repack.cpp:368-438`; `ime_kernels.h:82-91`; 237 new lines in
  `ime1_kernels.cpp`); `MUL_MAT_ID` rows with no expert zeroed instead of asserted (`U ime.cpp:685-691`); the 16-bit
  branch of `permute_transpose_impl` now calls the 16-bit transpose (`U rvv_kernels.cpp:642`; the provider does not
  carry that function, its `rvv_transposed_s16_mn_to_nm` is ggml-spacemit's attention copy); the buffer-type table gains
  `alloc_buffer_n` and `get_alloc_size_n` for API 3 (`U ime.cpp:1671-1678`). Nothing on question 8's items (the Q4_1
  zero point, the 1-row wrap, TCM recovery). Upstream also gained `docs/build-riscv64-spacemit.md` (SpacemiT's
  toolchain and QEMU builds). Relevant to D6 when the next kernel is copied, and to question 8's report. SpacemiT's fork, read later the same day (`PROJECT.md` item 26): one fix to a copied function, the Q4_0 scale fix of its PR #30, sits in the fork's in-tree copy only (taken, with a power-of-two factor); its other changes are new capability.
- **API version 3.** A FlagOS backend built on the mentor's tree (API 2) cannot be loaded as a module by today's
  upstream, nor the reverse (`U ggml-backend-reg.cpp:246`); E4's module experiment holds only on `ba360ef` until the
  tree is rebased.
- **Metal's `is_host` false even for shared and mapped buffers** (§3.5) is the upstream answer to "a unified-memory
  GPU backend": the CPU backend does not run ops on its buffers; the scheduler copies for CPU fallbacks; zero copy is
  the mapped model file. Our provider is more host-integrated than Metal already (it reads CPU buffers in place;
  Metal's `supports_buft` accepts only its own three types, `U ggml-metal.cpp:762-770`).
