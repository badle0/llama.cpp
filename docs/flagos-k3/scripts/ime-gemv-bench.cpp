// 1-row matmul bandwidth (plan.md): the upstream IME path's time for one MUL_MAT with one activation row, through
// ggml-cpu's CPU_RISCV64_SPACEMIT extra buffer type (the `ime` mode of build.md §7), at every 1-row matmul shape of a
// Qwen3-4B token, and the per-token total weighted as in `flagos-check-spacemit --bench` (lines "gemv").
// Also each result against ggml-cpu's plain path on the same weights, to show that the IME path ran and is right.
// Build on the K3 against the IME build (build-ime, GGML_CPU_RISCV64_SPACEMIT=ON), from the repository root, on one line:
//   g++ -O2 -std=c++17 -pthread -Iggml/include docs/flagos-k3/scripts/ime-gemv-bench.cpp build-ime/bin/libggml.so
//       build-ime/bin/libggml-base.so build-ime/bin/libggml-cpu.so -Wl,-rpath,$PWD/build-ime/bin -o ~/ime-gemv-bench
// Run: ~/tcmtest/tcmrelease --apply > /dev/null && OMP_THREAD_LIMIT=8 ~/ime-gemv-bench
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

struct shape {
    ggml_type type;
    int64_t   k, n;
    int       per_token;  // how often a Qwen3-4B token runs this shape
};

// bytes of one weight row in the IME layout: Q4_0 keeps its size, Q4_1 stores an 8-bit zero point instead of its fp16
// minimum, Q6_K is requantized to Q8_0
static double ime_row_bytes(ggml_type type, int64_t k) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return (double) (k / 32 * 18);
        case GGML_TYPE_Q4_1:
            return (double) (k / 32 * 19);
        default:
            return (double) (k / 32 * 34);
    }
}

// random weights in [-1, 1), quantized on 8 threads (the head has 389M of them)
static std::vector<uint8_t> random_weights(ggml_type type, int64_t k, int64_t n) {
    std::vector<uint8_t>     q(ggml_row_size(type, k) * n);
    std::vector<std::thread> workers;
    const int64_t            per = (n + 7) / 8;
    for (int t = 0; t < 8; t++) {
        const int64_t r0 = t * per, r1 = std::min(n, r0 + per);
        if (r0 >= r1) {
            break;
        }
        workers.emplace_back([&, t, r0, r1] {
            std::mt19937                          rng(1234 + t);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float>                    w((r1 - r0) * k);
            for (float & v : w) {
                v = dist(rng);
            }
            ggml_quantize_chunk(type, w.data(), q.data() + r0 * ggml_row_size(type, k), 0, r1 - r0, k, nullptr);
        });
    }
    for (std::thread & w : workers) {
        w.join();
    }
    return q;
}

static double nmse(const float * out, const float * ref, int64_t n) {
    double err = 0.0, norm = 0.0;
    for (int64_t i = 0; i < n; i++) {
        err += ((double) out[i] - ref[i]) * ((double) out[i] - ref[i]);
        norm += (double) ref[i] * ref[i];
    }
    return std::isfinite(err) && norm > 0.0 ? err / norm : HUGE_VAL;
}

int main() {
    ggml_backend_reg_t         reg       = ggml_backend_cpu_reg();
    ggml_backend_dev_t         dev       = ggml_backend_reg_dev_get(reg, 0);
    auto                       get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t ime       = nullptr;
    for (ggml_backend_buffer_type_t * b = get_extra != nullptr ? get_extra(dev) : nullptr; b != nullptr && *b != nullptr;
         b++) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_RISCV64_SPACEMIT") == 0) {
            ime = *b;
        }
    }
    if (ime == nullptr) {
        std::fprintf(stderr, "no CPU_RISCV64_SPACEMIT buffer type: link the IME build (GGML_CPU_RISCV64_SPACEMIT=ON)\n");
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);

    const shape shapes[] = {
        { GGML_TYPE_Q4_0, 2560, 4096, 36 }, { GGML_TYPE_Q4_0, 2560, 1024, 72 }, { GGML_TYPE_Q4_0, 4096, 2560, 36 },
        { GGML_TYPE_Q4_0, 2560, 9728, 72 }, { GGML_TYPE_Q4_0, 9728, 2560, 32 }, { GGML_TYPE_Q4_1, 9728, 2560, 4 },
        { GGML_TYPE_Q6_K, 2560, 151936, 1 },
    };
    double token_us = 0.0, token_bytes = 0.0;
    for (const shape & s : shapes) {
        // the weight twice: in the IME buffer type (repacked when set) and in a plain CPU buffer (ggml-cpu's own path)
        ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2, nullptr, true };
        ggml_context *   cw     = ggml_init(params);
        ggml_context *   cx     = ggml_init(params);
        ggml_tensor *    w_ime  = ggml_new_tensor_2d(cw, s.type, s.k, s.n);
        ggml_tensor *    w_cpu  = ggml_new_tensor_2d(cx, s.type, s.k, s.n);
        ggml_tensor *    x      = ggml_new_tensor_2d(cx, GGML_TYPE_F32, s.k, 1);
        ggml_tensor *    y_ime  = ggml_mul_mat(cx, w_ime, x);
        ggml_tensor *    y_cpu  = ggml_mul_mat(cx, w_cpu, x);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ime);
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
        if (bw == nullptr || bx == nullptr) {
            std::fprintf(stderr, "allocation failed for %s %lldx%lld\n", ggml_type_name(s.type), (long long) s.k,
                         (long long) s.n);
            return 1;
        }
        {
            const std::vector<uint8_t> q = random_weights(s.type, s.k, s.n);
            ggml_backend_tensor_set(w_ime, q.data(), 0, q.size());
            ggml_backend_tensor_set(w_cpu, q.data(), 0, q.size());
        }
        std::vector<float>                    xv(s.k);
        std::mt19937                          rng(7);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (float & v : xv) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));

        ggml_cgraph * g_ime = ggml_new_graph(cx);
        ggml_cgraph * g_cpu = ggml_new_graph(cx);
        ggml_build_forward_expand(g_ime, y_ime);
        ggml_build_forward_expand(g_cpu, y_cpu);
        const int iters = s.n > 100000 ? 5 : 30;
        for (int i = 0; i < 3; i++) {
            ggml_backend_graph_compute(cpu, g_ime);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            ggml_backend_graph_compute(cpu, g_ime);
        }
        const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        ggml_backend_graph_compute(cpu, g_cpu);

        const double bytes = (double) s.n * ime_row_bytes(s.type, s.k);
        token_us += t * s.per_token;
        token_bytes += bytes * s.per_token;
        std::printf("ime      gemv %-4s %5lldx%-6lld %8.0f us %6.1f GB/s; vs ggml-cpu's plain path NMSE %.2e%s\n",
                    ggml_type_name(s.type), (long long) s.k, (long long) s.n, t, bytes / t / 1e3,
                    nmse(static_cast<const float *>(y_ime->data), static_cast<const float *>(y_cpu->data), s.n),
                    w_ime->extra != nullptr ? "" : " (no IME traits on the weight: not the IME path)");
        ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bx);
        ggml_free(cw);
        ggml_free(cx);
    }
    std::printf("ime      gemv per Qwen3-4B token (%.0f MB of weights) %6.1f ms %6.1f GB/s\n", token_bytes / 1e6,
                token_us / 1e3, token_bytes / token_us / 1e3);
    ggml_backend_free(cpu);
    return 0;
}
