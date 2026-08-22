# FlagOS AMD provider boundary

The AMD provider must implement `flagos_amd_provider()` from
`../../flagos-provider.h`. It is a peer of the Denglin provider, not a branch in
the Denglin launcher and not a wrapper around the upstream `ggml-hip` backend.

The first target is the Radeon 890M in Ryzen AI 9 HX 370 (`gfx1150`). Keep all
HIP and ROCm types inside this directory. The provider-neutral registry and
graph planner must continue to build with no ROCm installation.

The provider owns:

- HIP device discovery and a stable UUID or PCI-derived identity;
- device, pinned, and coherent host-visible memory where the APU supports it;
- HIP streams, events, copies, synchronization, and backend lifetime;
- HSACO loading, symbol lookup, packed argument ABI, and kernel launch;
- hipBLAS or rocBLAS dispatch and any provider-specific quantized library;
- `supports_op` validation matching the actual AOT package;
- lowering of accepted common graph patterns;
- HIP Graph capture, instantiate, replay, invalidation, and bounded caching.

The implementation must return a full GGML device from `device_get()`. The
common registry assigns its global FlagOS ordinal and preserves the provider's
local ordinal. Device names must be globally unambiguous, for example
`FlagOS:AMD:0`. Provider IDs and device UUIDs must be nonzero and unique.

Suggested source layout:

```text
providers/amd/
    provider.cmake
    flagos-amd.cpp
    flagos-amd-api.h
    kernels/
        manifest.json
        gfx1150/*.hsaco
```

`provider.cmake` should discover HIP and the selected vendor libraries only
when `GGML_FLAGOS_AMD=ON`, append the AMD sources and libraries to the parent
`FLAGOS_*` lists, and define `GGML_FLAGOS_HAVE_AMD`. Replace its current
fail-closed diagnostic only after the provider source and its conformance tests
exist.

Minimum enablement gates:

1. `GGML_FLAGOS_DENGLIN=OFF` and `GGML_FLAGOS_AMD=ON` configure on a ROCm host.
2. Core and Graph Plan tests pass without linking Denglin libraries.
3. Device enumeration, allocation, copies, events, and synchronization pass.
4. Every advertised op passes `test-backend-ops` numerically.
5. Qwen3-4B Q4_K_M runs end to end with unsupported nodes on CPU.
6. HIP Graph reports real capture and replay counters with output unchanged.
7. Cold-start, TTFT, decode throughput, memory, and fallback counts are recorded.
