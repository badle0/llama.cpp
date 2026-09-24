# ggml-flagos design review (mentor's `feature/flagos-amd-890-opt` @ `5794e12`)

Line numbers refer to `5794e12`. Paths are relative to `ggml/src/ggml-flagos/` unless given in full. "Design" = `ggml-flagos-provider-design.pdf` V2.0.

## What is sound (keep)

- Scheduler stays authoritative for placement/splits/copies; providers act inside `graph_compute(split)`. No scheduler patch needed.
- `supports_op` = "this node can run here alone, correctly" (Design §7). AMD's op-by-op checks follow it.
- No fallback after a visible write; fail the graph instead (Design §3.3, §12.3).
- Fused-only pre-placement partitioning deferred until hardware evidence (Design §11).

## A. Fusion and the provider shim (mentor's focus)

Current flow: Common `flagos_enumerate_candidates()` (`flagos-graph-plan.cpp` L785–1265) → provider `query_lowering(cgraph, candidate)` → greedy non-overlap plan → provider `execute_fusion(cgraph, step)`. A provider sees a pattern ID, `node_indices`, and `required_output_node_indices`; it can only accept or reject. Denglin accepts only `rms_norm_mul` (`providers/denglin/flagos-denglin.cpp` L1828–1876); AMD handles most of the 16 patterns.

A1. **Closed, Common-owned pattern catalog.** Patterns are hand-written matchers in Common; IDs are a global append-only enum (`flagos-graph-plan.h`). A provider cannot propose a fusion whose node set differs from an existing pattern (smaller, larger, or cut at another boundary, e.g. dequant+GEMM+epilogue tiled for TCM). Adding one edits Common for every provider. Several patterns (`gated_delta_net_*`, `ssm_conv_silu`, `gdn_gate_projections`) come from AMD's Qwen3.5 tuning.

A2. **Pattern layouts are an implicit positional ABI.** The meaning of `node_indices[i]` exists only in matcher code. Providers decode by position (e.g. `providers/amd/flagos-amd.cpp` L3601–3603; 67 positional uses in AMD). One ID can have several layouts: `attention_output_gate` is `{gate, mul}` or `{add, gate, mul}` (`flagos-graph-plan.cpp` ~L1063 vs ~L1085); AMD distinguishes them by count (L3685 vs L3697). Append-only IDs protect numbers, not layouts.

A3. **Semantic validation duplicated per provider; Common's safety checks encode one provider's assumptions.** Each provider re-checks ops, shapes, contiguity, overlap. Denglin re-runs overlap checks because Common admits an in-place layout AMD's kernel handles (comment `flagos-denglin.cpp` L1853–1858). Cost grows with patterns × providers.

A4. **No shared pattern→kernel binding.** `execute_fusion` receives raw `ggml_cgraph*`; each provider maps tensors to kernel args by hand (Denglin `launch_rms_norm_mul(...)` L1888). AOT manifests describe kernel args, but nothing links pattern tensors to them. Even when two chips compile the same FlagGems/Triton kernel, each provider rewrites the binding — undermining "one kernel source, many chips".

A5. **Fusion happens after per-node placement.** A pattern straddling a CPU/FlagOS boundary is lost, and every member must also be directly supported (Design §7.1): a fused `rope_kv_store` kernel is unusable without a standalone `SET_ROWS` kernel. Fine for GPU providers that claim nearly everything; fragments fusion for providers with partial coverage (the K3 during bring-up). Design §11 covers only the fused-only extreme.

A6. **The shim is three unconnected mechanisms.** `flagos_provider_v1` (discovery only), ggml vtables (execution), `flagos_fusion_interface` (built ad hoc inside `graph_compute`: AMD L4748, Denglin L2261). The descriptor does not declare lowerable patterns, implementation IDs, or required AOT kernels, so Common cannot report coverage, build test matrices, or detect a package missing a kernel. `implementation_id` values are provider-private magic numbers (AMD GDN: 12/13/14, L3835–3839).

Directions to discuss (options, not decisions):
- Pattern descriptors as data with **named roles**; Common emits role bindings instead of positional indices (fixes A2, reduces A3).
- **Provider-proposed candidates** within a split, validated by Common's safety checks (fixes A1).
- **Declare capabilities** in the descriptor (patterns, implementation IDs, required kernels), cross-checked against the AOT manifest at load (A6).
- **Shared role→argument binding** for kernels generated from the same FlagGems source (A4).
- Fusion-aware placement needs scheduler changes (A5); interim rule: implement direct kernels for every member op of a pattern you want to fuse.

K3 relevance: the K3's valuable fused structures (IME matmul with dequant/epilogue, TCM-tiled blocks) likely don't match the current catalog — a concrete test case for A1.

## B. Other structural issues

B1. **Out-of-tree provider registration is unreachable from llama.cpp.** FlagOS registers inside ggml's global registry constructor (`ggml/src/ggml-backend-reg.cpp` L127–128), which llama.cpp triggers at startup; registration then closes (`flagos-registry.cpp` L146). No code outside `ggml-flagos/` calls `flagos_registry_register_provider`. With the C++ source ABI (Design §5.1), providers are compile-time only (`flagos_compiled_providers()`, L69–78); the loadable unit is `libggml-flagos` as a whole.

B2. **"Select exactly one provider" is not enforced on llama.cpp's path.** Nothing in llama.cpp calls `flagos_select_provider_v1` or `ggml_backend_flagos_init`; the registry exposes every device of every compiled provider (L99–103). In multi-provider builds llama.cpp's own device logic decides (GPU layer split, ACCEL priority), unless the user restricts with `--device`.

B3. **GPU-shaped execution model; shared-memory CPU accelerators unaddressed.** `memory_domain_id` is only copied into the profile (`flagos-provider.cpp` L113), never used to avoid copies between same-memory devices. Default fusion score counts launches/bytes; on spine-runtime (one launch per split, per-node barriers) the costs are barriers and TCM residency. Nothing on core co-scheduling with the CPU backend or TCM ownership. Splits run sequentially (`ggml/src/ggml-backend.cpp` L1731), so little CPU/accelerator concurrency in practice.

B4. **Thin Common; each provider is a full backend.** Providers implement device/buffer/backend/stream themselves (AMD 6,201 lines, Denglin 3,223) and each has its own AOT manifest loader (`providers/amd/flagos-amd-aot.cpp`, `flagos-denglin.cpp`); the common package schema/validator (Design §9.4) is roadmap only (§15.1). For the K3 this overlaps with SpacemiT's `ggml-spacemit` (a full ACCEL backend on spine-runtime) — scope question for the mentor.

B5. **Fail-closed ⇒ failed requests (deliberate tradeoff).** Any post-placement provider error fails `llama_decode`; robustness depends on exact `supports_op`.

## C. Code-level findings

- `GGML_FLAGOS_DENGLIN` defaults to ON (`ggml/CMakeLists.txt` L202) and hard-fails without the Denglin SDK; not mentioned in the Design.
- macOS link failure: `ggml_backend_flagos_reg_devices()` called `ggml_backend_register` (in `libggml`) from the backend library; fixed on `feature/flagos-spacemit` (`eaa25ff`). Candidate upstream fix for the mentor.
- Design is an AI-written summary; §14 numbers and packages live outside the repo (Design §16.2) and can't be verified from code.
