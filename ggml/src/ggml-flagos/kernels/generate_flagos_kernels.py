#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from merge_flagos_module import merge as merge_llvm_modules

# Triton's dlgpu backend compiles one kernel per cubin. These are the flags it
# passes to dlcc in make_cubin; the merged module has to be built the same way.
DLCC_TARGET = "dlgput64-unknown-cuda"
DLCC_OPTIONS = [
    "-ftz",
    "-mllvm", "-dlgpu-lower-ptx=true",
    "-Xllc", "--dlgpu-full-shfl=true",
    "-Xllc", "--fp-contract=fast",
    "-soft-spill-allocator",
]
MERGED_MODULE_NAME = "flagos_kernels.cubin"


BLOCK_SIZE = 256
NUM_WARPS = 4
# SSM_CONV+SiLU is launch-bound for single-token decode. Providers may tune
# this independently from the generic elementwise launch without changing the
# kernel ABI. The common Denglin package keeps the historical 256x4 default.
SSM_CONV_SILU_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_SSM_CONV_SILU_BLOCK_SIZE", str(BLOCK_SIZE)))
SSM_CONV_SILU_NUM_WARPS = int(os.environ.get(
    "FLAGOS_SSM_CONV_SILU_NUM_WARPS", str(NUM_WARPS)))
# Decode GEMV assigns one program to one output row. KS20-A was tuned with one
# warp; keep this target-selectable so providers can benchmark a different
# resident-lane count without changing the common kernel ABI.
GEMV_NUM_WARPS = int(os.environ.get("FLAGOS_GEMV_NUM_WARPS", "1"))
# Must be >= the widest row we accept in supports_op. Qwen3.5-9B has a
# 4096-wide hidden state; at 2048 every RMS norm was declined to the CPU.
RMS_NORM_BLOCK_SIZE = 4096
# Columns processed per program by the batched (prefill) quantized GEMMs.
# The dequantized weight row is reused across the whole tile.
MUL_MAT_COLS_PER_BLOCK = 16
# Experimental tiled quantized prefill kernels.  These use MFMA-friendly
# tl.dot tiles after decoding a small K slice, avoiding the one-output-element
# program used by the legacy batched path.  Keep the shape explicit in the
# manifest so providers can tune it per target without changing the ABI.
QUANT_TILE_BLOCK_M = int(os.environ.get("FLAGOS_QUANT_TILE_BLOCK_M", "16"))
QUANT_TILE_BLOCK_N = int(os.environ.get("FLAGOS_QUANT_TILE_BLOCK_N", "32"))
QUANT_TILE_NUM_WARPS = int(os.environ.get("FLAGOS_QUANT_TILE_NUM_WARPS", "4"))
# Row-wise kernels use one program per row with BLOCK spanning the whole row.
ROW_BLOCK_SIZE = 1024
QK_K = 256
QK4_0 = 32
Q4_0_BLOCK_BYTES = 18
QK4_1 = 32
Q4_1_BLOCK_BYTES = 20
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34
Q5_K_BLOCK_BYTES = 176
Q4_K_BLOCK_BYTES = 144
Q6_K_BLOCK_BYTES = 210
ATTENTION_HEAD_DIM = 128
# Compile one runtime-GQA kernel and validate both the original Qwen2.5 shape
# and Qwen3-4B.  Keeping the head mapping out of constexpr avoids one AOT
# binary per model/head configuration.
ATTENTION_TEST_CONFIGS = ((12, 2), (32, 8))
ATTENTION_BLOCK_M = 16
ATTENTION_BLOCK_N = 32


@triton.jit
def flagos_add_f32(x, y, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flagos_add_repeat_f32(x, y, output, n_elements, y_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets % y_elements, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flagos_mul_f32(x, y, output, n_elements, y_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y_offsets = offsets % y_elements
    values = tl.load(x + offsets, mask=mask) * tl.load(y + y_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flagos_get_rows_q4_0_f32(weights_u8, weights_f16, row_index, output, n_cols):
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 32
    block = row * blocks_per_row + sub
    lanes = tl.arange(0, 32)
    block_byte = block * 18
    d = tl.load(weights_f16 + block * 9).to(tl.float32)
    quant_byte = tl.load(weights_u8 + block_byte + 2 + lanes % 16)
    quant = tl.where(lanes < 16, quant_byte & 15, quant_byte >> 4).to(tl.int32) - 8
    tl.store(output + token * n_cols + sub * 32 + lanes,
             d * quant.to(tl.float32))


@triton.jit
def flagos_get_rows_q4_1_f32(weights_u8, weights_f16, row_index, output, n_cols):
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 32
    block = row * blocks_per_row + sub
    lanes = tl.arange(0, 32)
    block_byte = block * 20
    d = tl.load(weights_f16 + block * 10).to(tl.float32)
    m = tl.load(weights_f16 + block * 10 + 1).to(tl.float32)
    quant_byte = tl.load(weights_u8 + block_byte + 4 + lanes % 16)
    quant = tl.where(lanes < 16, quant_byte & 15, quant_byte >> 4).to(tl.float32)
    tl.store(output + token * n_cols + sub * 32 + lanes, d * quant + m)


@triton.jit
def flagos_get_rows_q8_0_f32(weights_u8, weights_f16, row_index, output, n_cols):
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 32
    block = row * blocks_per_row + sub
    lanes = tl.arange(0, 32)
    block_byte = block * 34
    d = tl.load(weights_f16 + block * 17).to(tl.float32)
    quant = tl.load(weights_u8 + block_byte + 2 + lanes).to(tl.int8).to(tl.float32)
    tl.store(output + token * n_cols + sub * 32 + lanes, d * quant)


@triton.jit
def flagos_get_rows_q5_k_f32(weights_u8, weights_f16, row_index, output, n_cols):
    # Q5_K has the Q4_K scale/min layout plus one high-bit plane per 32-lane
    # sub-block.  Keep the byte decode identical to ggml's dequantize_row_q5_K.
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 256
    block = row * blocks_per_row + sub
    lanes = tl.arange(0, 256)
    block_byte = block * 176
    block_half = block_byte // 2
    d = tl.load(weights_f16 + block_half).to(tl.float32)
    dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

    group = lanes // 32
    scale_lo_index = tl.where(group < 4, group, group + 4)
    scale_lo = tl.load(weights_u8 + block_byte + 4 + scale_lo_index)
    scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
    scale = tl.where(
        group < 4,
        scale_lo & 63,
        (scale_lo & 15) | ((scale_hi >> 6) << 4),
    ).to(tl.float32)

    min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
    min_hi = tl.load(weights_u8 + block_byte + 4 + group)
    minimum = tl.where(
        group < 4,
        min_lo & 63,
        (min_lo >> 4) | ((min_hi >> 6) << 4),
    ).to(tl.float32)

    within_64 = lanes % 64
    quant_byte = tl.load(
        weights_u8 + block_byte + 48 + (lanes // 64) * 32 + within_64 % 32
    )
    low = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4)
    high_byte = tl.load(weights_u8 + block_byte + 16 + lanes % 32)
    high = (high_byte >> (group % 8)) & 1
    quant = (low | (high << 4)).to(tl.float32)
    tl.store(output + token * n_cols + sub * 256 + lanes,
             d * scale * quant - dmin * minimum)


@triton.jit
def flagos_get_rows_q4_k_f32(weights_u8, weights_f16, row_index, output, n_cols):
    # GET_ROWS on a Q4_K table: dequantize only the requested rows.
    # One program per (row, 256-element super-block) pair, mirroring the layout
    # decode in flagos_dequant_q4_k_f16.
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 256
    block = row * blocks_per_row + sub

    lanes = tl.arange(0, 256)
    block_byte = block * 144
    block_half = block_byte // 2
    d = tl.load(weights_f16 + block_half).to(tl.float32)
    dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

    group = lanes // 32
    scale_lo_index = tl.where(group < 4, group, group + 4)
    scale_lo = tl.load(weights_u8 + block_byte + 4 + scale_lo_index)
    # Clamped for the same reason as in the mul_mat kernels: tl.where evaluates
    # both branches, so a negative index here reads outside the tensor.
    scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
    scale = tl.where(
        group < 4,
        scale_lo & 63,
        (scale_lo & 15) | ((scale_hi >> 6) << 4),
    ).to(tl.float32)

    min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
    min_hi = tl.load(weights_u8 + block_byte + 4 + group)
    minimum = tl.where(
        group < 4,
        min_lo & 63,
        (min_lo >> 4) | ((min_hi >> 6) << 4),
    ).to(tl.float32)

    within_64 = lanes % 64
    quant_byte = tl.load(
        weights_u8 + block_byte + 16 + (lanes // 64) * 32 + within_64 % 32
    )
    quant = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4).to(tl.float32)
    tl.store(output + token * n_cols + sub * 256 + lanes,
             d * scale * quant - dmin * minimum)


@triton.jit
def flagos_get_rows_q6_k_f32(weights_u8, weights_f16, row_index, output, n_cols):
    # Qwen3-4B stores its token embedding table as Q6_K even in Q4_K_M files.
    token = tl.program_id(0)
    sub = tl.program_id(1)
    row = tl.load(row_index + token)

    blocks_per_row = n_cols // 256
    block = row * blocks_per_row + sub

    lanes = tl.arange(0, 256)
    block_byte = block * 210
    d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
    half = lanes // 128
    quadrant = (lanes % 128) // 32
    lane = lanes % 32

    ql_index = half * 64 + lane + (quadrant % 2) * 32
    ql_byte = tl.load(weights_u8 + block_byte + ql_index)
    ql = tl.where(quadrant < 2, ql_byte & 15, ql_byte >> 4)
    qh_byte = tl.load(weights_u8 + block_byte + 128 + half * 32 + lane)
    qh = (qh_byte >> (quadrant * 2)) & 3
    quant = (ql | (qh << 4)).to(tl.int32) - 32

    scale_index = half * 8 + quadrant * 2 + lane // 16
    scale_u8 = tl.load(weights_u8 + block_byte + 192 + scale_index)
    scale = scale_u8.to(tl.int8).to(tl.float32)
    tl.store(output + token * n_cols + sub * 256 + lanes,
             d * scale * quant.to(tl.float32))


@triton.jit
def flagos_get_rows_f32(table, row_index, output, n_cols, BLOCK: tl.constexpr):
    # Dense gather used for final-token selection around Qwen3's output graph.
    token = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    row = tl.load(row_index + token)
    values = tl.load(table + row * n_cols + cols, mask=mask, other=0.0)
    tl.store(output + token * n_cols + cols, values, mask=mask)


@triton.jit
def flagos_ssm_conv_f32(
        s, c, output,
        d_conv, d_inner, n_tokens, n_seqs,
        ss1, ss2,
        sc1,
        so0, so1,
        BLOCK: tl.constexpr):
    # Causal depthwise conv1d over the rolling state window (Qwen3.5 linear attn).
    # CPU reference: ggml-cpu/ops.cpp ggml_compute_forward_ssm_conv_f32.
    #   src0 = conv state [d_conv-1+n_tokens, d_inner, n_seqs]
    #   src1 = conv weight [d_conv, d_inner]
    #   dst  = [d_inner, n_tokens, n_seqs]
    # Each program handles a BLOCK-wide slice of channels for one (token, seq).
    token = tl.program_id(0)
    seq = tl.program_id(1)
    base = tl.program_id(2) * BLOCK
    channels = base + tl.arange(0, BLOCK)
    mask = channels < d_inner

    total = tl.zeros([BLOCK], dtype=tl.float32)
    for tap in range(d_conv):
        # State is indexed [time, channel]; the window for this token starts at
        # `token` and advances one step per tap.
        state = tl.load(s + seq * ss2 + channels * ss1 + (token + tap),
                        mask=mask, other=0.0)
        weight = tl.load(c + channels * sc1 + tap, mask=mask, other=0.0)
        total += state * weight

    tl.store(output + seq * so1 + token * so0 + channels, total, mask=mask)


@triton.jit
def flagos_ssm_conv_silu_f32(
        s, c, output,
        d_conv, d_inner, n_tokens, n_seqs,
        ss1, ss2,
        sc1,
        so0, so1,
        BLOCK: tl.constexpr):
    # SSM_CONV followed by SiLU. The graph planner proves that the unfused
    # convolution output has no external consumer before selecting this path.
    token = tl.program_id(0)
    seq = tl.program_id(1)
    base = tl.program_id(2) * BLOCK
    channels = base + tl.arange(0, BLOCK)
    mask = channels < d_inner

    total = tl.zeros([BLOCK], dtype=tl.float32)
    for tap in range(d_conv):
        state = tl.load(s + seq * ss2 + channels * ss1 + (token + tap),
                        mask=mask, other=0.0)
        weight = tl.load(c + channels * sc1 + tap, mask=mask, other=0.0)
        total += state * weight

    tl.store(output + seq * so1 + token * so0 + channels,
             total * tl.sigmoid(total), mask=mask)


@triton.jit
def flagos_gated_delta_net_scalar_f32(
        q, k, v, gate, beta, current_state, output,
        state_size, n_heads, n_tokens, n_seqs,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        q_heads, q_seq_ratio, snapshot_count, scale,
        BLOCK: tl.constexpr, COLS: tl.constexpr):
    head = tl.program_id(0)
    seq = tl.program_id(1)
    col_base = tl.program_id(2) * COLS
    cols = col_base + tl.arange(0, COLS)
    lanes = tl.arange(0, BLOCK)
    col_mask = cols < state_size
    lane_mask = lanes < state_size

    q_head = head % q_heads
    q_seq = seq // q_seq_ratio
    state_base = (seq * n_heads + head) * state_size * state_size
    state_offsets = state_base + cols[:, None] * state_size + lanes[None, :]
    state = tl.load(current_state + state_offsets,
                    mask=col_mask[:, None] & lane_mask[None, :], other=0.0)

    attention_elements = state_size * n_heads * n_tokens * n_seqs
    snapshot_stride = state_size * state_size * n_heads * n_seqs
    for token in tl.range(0, n_tokens):
        q_base = q_seq * sq3 + token * sq2 + q_head * sq1
        q_values = tl.load(q + q_base + lanes, mask=lane_mask, other=0.0)
        k_values = tl.load(k + q_base + lanes, mask=lane_mask, other=0.0)
        v_base = seq * sv3 + token * sv2 + head * sv1
        v_values = tl.load(v + v_base + cols, mask=col_mask, other=0.0)
        gate_beta_base = seq * sb3 + token * sb2 + head * sb1
        decay = tl.exp(tl.load(gate + gate_beta_base))
        beta_value = tl.load(beta + gate_beta_base)

        state *= decay
        projected_key = tl.sum(state * k_values[None, :], axis=1)
        delta = (v_values - projected_key) * beta_value
        state += delta[:, None] * k_values[None, :]
        attention = tl.sum(state * q_values[None, :], axis=1) * scale
        attention_offset = ((seq * n_tokens + token) * n_heads + head) * state_size
        tl.store(output + attention_offset + cols, attention, mask=col_mask)

        snapshot = n_tokens - 1 - token
        snapshot_offsets = (attention_elements + snapshot * snapshot_stride +
                            state_base + cols[:, None] * state_size + lanes[None, :])
        tl.store(output + snapshot_offsets, state,
                 mask=(snapshot < snapshot_count) & col_mask[:, None] & lane_mask[None, :])


@triton.jit
def flagos_gated_delta_net_scalar_f32_cache(
        q, k, v, gate, beta, current_state, output, cache,
        state_size, n_heads, n_tokens, n_seqs,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        q_heads, q_seq_ratio, snapshot_count, scale, cache_slot_stride,
        BLOCK: tl.constexpr, COLS: tl.constexpr):
    head = tl.program_id(0)
    seq = tl.program_id(1)
    col_base = tl.program_id(2) * COLS
    cols = col_base + tl.arange(0, COLS)
    lanes = tl.arange(0, BLOCK)
    col_mask = cols < state_size
    lane_mask = lanes < state_size

    q_head = head % q_heads
    q_seq = seq // q_seq_ratio
    state_base = (seq * n_heads + head) * state_size * state_size
    state_offsets = state_base + cols[:, None] * state_size + lanes[None, :]
    state = tl.load(current_state + state_offsets,
                    mask=col_mask[:, None] & lane_mask[None, :], other=0.0)

    attention_elements = state_size * n_heads * n_tokens * n_seqs
    snapshot_stride = state_size * state_size * n_heads * n_seqs
    for token in tl.range(0, n_tokens):
        q_base = q_seq * sq3 + token * sq2 + q_head * sq1
        q_values = tl.load(q + q_base + lanes, mask=lane_mask, other=0.0)
        k_values = tl.load(k + q_base + lanes, mask=lane_mask, other=0.0)
        v_base = seq * sv3 + token * sv2 + head * sv1
        v_values = tl.load(v + v_base + cols, mask=col_mask, other=0.0)
        gate_beta_base = seq * sb3 + token * sb2 + head * sb1
        decay = tl.exp(tl.load(gate + gate_beta_base))
        beta_value = tl.load(beta + gate_beta_base)

        state *= decay
        projected_key = tl.sum(state * k_values[None, :], axis=1)
        delta = (v_values - projected_key) * beta_value
        state += delta[:, None] * k_values[None, :]
        attention = tl.sum(state * q_values[None, :], axis=1) * scale
        attention_offset = ((seq * n_tokens + token) * n_heads + head) * state_size
        tl.store(output + attention_offset + cols, attention, mask=col_mask)

        snapshot = n_tokens - 1 - token
        snapshot_mask = ((snapshot < snapshot_count) & col_mask[:, None] &
                         lane_mask[None, :])
        output_offsets = (attention_elements + snapshot * snapshot_stride +
                          state_base + cols[:, None] * state_size + lanes[None, :])
        cache_offsets = (snapshot * cache_slot_stride + state_base +
                         cols[:, None] * state_size + lanes[None, :])
        tl.store(output + output_offsets, state, mask=snapshot_mask)
        tl.store(cache + cache_offsets, state, mask=snapshot_mask)


@triton.jit
def flagos_gated_delta_net_scalar_f32_cache_only(
        q, k, v, gate, beta, current_state, output, cache,
        state_size, n_heads, n_tokens, n_seqs,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        q_heads, q_seq_ratio, snapshot_count, scale, cache_slot_stride,
        BLOCK: tl.constexpr, COLS: tl.constexpr,
        EXACT_STATE_SIZE: tl.constexpr, EXACT_SCALE: tl.constexpr):
    head = tl.program_id(0)
    seq = tl.program_id(1)
    col_base = tl.program_id(2) * COLS
    cols = col_base + tl.arange(0, COLS)
    lanes = tl.arange(0, BLOCK)
    q_head = head % q_heads
    q_seq = seq // q_seq_ratio
    state_base = (seq * n_heads + head) * EXACT_STATE_SIZE * EXACT_STATE_SIZE
    state_offsets = state_base + cols[:, None] * EXACT_STATE_SIZE + lanes[None, :]
    state = tl.load(current_state + state_offsets)

    for token in tl.range(0, n_tokens):
        q_base = q_seq * sq3 + token * sq2 + q_head * sq1
        q_values = tl.load(q + q_base + lanes)
        k_values = tl.load(k + q_base + lanes)
        v_base = seq * sv3 + token * sv2 + head * sv1
        v_values = tl.load(v + v_base + cols)
        gate_beta_base = seq * sb3 + token * sb2 + head * sb1
        decay = tl.exp(tl.load(gate + gate_beta_base))
        beta_value = tl.load(beta + gate_beta_base)

        state *= decay
        projected_key = tl.sum(state * k_values[None, :], axis=1)
        delta = (v_values - projected_key) * beta_value
        state += delta[:, None] * k_values[None, :]
        attention = tl.sum(state * q_values[None, :], axis=1) * EXACT_SCALE
        attention_offset = ((seq * n_tokens + token) * n_heads + head) * EXACT_STATE_SIZE
        tl.store(output + attention_offset + cols, attention)

    tl.store(cache + state_offsets, state)


@triton.jit
def flagos_gated_delta_net_scalar_f32_cache_only_decode(
        q, k, v, gate, beta, current_state, output, cache,
        state_size, n_heads, n_tokens, n_seqs,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        q_heads, q_seq_ratio, snapshot_count, scale, cache_slot_stride,
        BLOCK: tl.constexpr, COLS: tl.constexpr,
        EXACT_STATE_SIZE: tl.constexpr, EXACT_SCALE: tl.constexpr):
    """Exact one-token lowering for the cache-only recurrent path.

    The public ABI deliberately matches the generic cache-only kernel so the
    provider can substitute it without changing the graph contract.  Runtime
    lowering must prove n_tokens == snapshot_count == 1, an exact state size,
    a power-of-two q_heads value, and a one-to-one Q/V sequence mapping before
    dispatch. The unused ABI values remain present so the HSACO call boundary
    stays uniform. Multiple independent sequences remain valid.
    """
    head = tl.program_id(0)
    seq = tl.program_id(1)
    col_base = tl.program_id(2) * COLS
    cols = col_base + tl.arange(0, COLS)
    lanes = tl.arange(0, BLOCK)
    q_head = head & (q_heads - 1)
    q_seq = seq
    state_base = (seq * n_heads + head) * EXACT_STATE_SIZE * EXACT_STATE_SIZE
    state_offsets = state_base + cols[:, None] * EXACT_STATE_SIZE + lanes[None, :]
    state = tl.load(current_state + state_offsets)

    q_base = q_seq * sq3 + q_head * sq1
    q_values = tl.load(q + q_base + lanes)
    k_values = tl.load(k + q_base + lanes)
    v_base = seq * sv3 + head * sv1
    v_values = tl.load(v + v_base + cols)
    gate_beta_base = seq * sb3 + head * sb1
    decay = tl.exp(tl.load(gate + gate_beta_base))
    beta_value = tl.load(beta + gate_beta_base)

    state *= decay
    projected_key = tl.sum(state * k_values[None, :], axis=1)
    delta = (v_values - projected_key) * beta_value
    state += delta[:, None] * k_values[None, :]
    attention = tl.sum(state * q_values[None, :], axis=1) * EXACT_SCALE
    attention_offset = (seq * n_heads + head) * EXACT_STATE_SIZE
    tl.store(output + attention_offset + cols, attention)
    tl.store(cache + state_offsets, state)


@triton.jit(do_not_specialize=[
    "xe0", "xe1", "xe2", "xe3",
    "ye0", "ye1", "ye2", "ye3",
    "sx0", "sx1", "sx2", "sx3",
    "sy0", "sy1", "sy2", "sy3",
    "n_elements",
])
def flagos_copy_strided_f32(
        x, output,
        xe0, xe1, xe2, xe3,
        ye0, ye1, ye2, ye3,
        sx0, sx1, sx2, sx3,
        sy0, sy1, sy2, sy3,
        n_elements,
        BLOCK: tl.constexpr):
    # Generic 4D strided copy. Matches the CPU dup_bytes reference
    # (ggml-cpu/ops.cpp:372): CPY pairs elements by FLAT ORDER, i.e. the n-th
    # element of the source in its own row-major traversal goes to the n-th
    # position of the destination in ITS row-major traversal. So each side is
    # decomposed with its OWN shape and then indexed with its own strides.
    # For same-shape permutations both decompositions coincide, so this also
    # covers the permutation case; it additionally handles flattening reshapes
    # such as [3,8192]->[24576] where the shapes differ.
    # Strides are in elements (caller divides nb[] by sizeof(float)).
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Source coordinates from the source shape.
    a0 = offsets % xe0
    ra = offsets // xe0
    a1 = ra % xe1
    ra = ra // xe1
    a2 = ra % xe2
    a3 = ra // xe2

    # Destination coordinates from the destination shape.
    b0 = offsets % ye0
    rb = offsets // ye0
    b1 = rb % ye1
    rb = rb // ye1
    b2 = rb % ye2
    b3 = rb // ye2

    src = a0 * sx0 + a1 * sx1 + a2 * sx2 + a3 * sx3
    dst = b0 * sy0 + b1 * sy1 + b2 * sy2 + b3 * sy3

    tl.store(output + dst, tl.load(x + src, mask=mask), mask=mask)


@triton.jit
def flagos_copy_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.load(x + offsets, mask=mask), mask=mask)


@triton.jit
def flagos_concat_f32(
        a, b, output, dim,
        ye0, ye1, ye2, ye3,
        ae0, ae1, ae2, ae3,
        as0, as1, as2, as3,
        bs0, bs1, bs2, bs3,
        n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    y0 = offsets % ye0
    rest = offsets // ye0
    y1 = rest % ye1
    rest = rest // ye1
    y2 = rest % ye2
    y3 = rest // ye2

    a_extent = tl.where(dim == 0, ae0,
                        tl.where(dim == 1, ae1, tl.where(dim == 2, ae2, ae3)))
    coord = tl.where(dim == 0, y0, tl.where(dim == 1, y1, tl.where(dim == 2, y2, y3)))
    from_a = valid & (coord < a_extent)
    b0 = tl.where(dim == 0, y0 - a_extent, y0)
    b1 = tl.where(dim == 1, y1 - a_extent, y1)
    b2 = tl.where(dim == 2, y2 - a_extent, y2)
    b3 = tl.where(dim == 3, y3 - a_extent, y3)
    a_offset = y0 * as0 + y1 * as1 + y2 * as2 + y3 * as3
    b_offset = b0 * bs0 + b1 * bs1 + b2 * bs2 + b3 * bs3
    value_a = tl.load(a + a_offset, mask=from_a, other=0.0)
    value_b = tl.load(b + b_offset, mask=valid & ~from_a, other=0.0)
    tl.store(output + offsets, value_a + value_b, mask=valid)


@triton.jit
def flagos_scale_f32(x, output, scale, bias, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask) * scale + bias
    tl.store(output + offsets, values, mask=mask)


# Binary elementwise ops with ggml's trailing-dimension broadcast: src1 is indexed
# modulo its own element count, matching flagos_mul_f32 above.
@triton.jit
def flagos_sub_f32(x, y, output, n_elements, y_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y_values = tl.load(y + offsets % y_elements, mask=mask)
    tl.store(output + offsets, tl.load(x + offsets, mask=mask) - y_values, mask=mask)


@triton.jit
def flagos_div_f32(x, y, output, n_elements, y_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y_values = tl.load(y + offsets % y_elements, mask=mask)
    tl.store(output + offsets, tl.load(x + offsets, mask=mask) / y_values, mask=mask)


# Unary activations used by the gated delta net gate path.
@triton.jit
def flagos_sigmoid_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.sigmoid(tl.load(x + offsets, mask=mask)), mask=mask)


@triton.jit
def flagos_exp_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.exp(tl.load(x + offsets, mask=mask)), mask=mask)


@triton.jit
def flagos_softplus_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask)
    # log1p(exp(v)) overflows for large v; fall back to the identity above the
    # threshold where softplus(v) == v to within f32 precision.
    result = tl.where(values > 20.0, values, tl.log(1.0 + tl.exp(values)))
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def flagos_fill_f32(output, value, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.full((BLOCK,), value, tl.float32), mask=mask)


# One program per row: sum and L2-normalise along ne[0].
@triton.jit
def flagos_sum_rows_f32(x, output, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, 0.0)
    tl.store(output + row, tl.sum(values, axis=0))


@triton.jit
def flagos_l2_norm_f32(x, output, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, 0.0)
    # ggml clamps the root, not the sum: scale = 1/max(sqrt(sum), eps).
    scale = 1.0 / tl.maximum(tl.sqrt(tl.sum(values * values, axis=0)), eps)
    tl.store(output + row * n_cols + offsets, values * scale, mask=mask)


@triton.jit
def flagos_l2_norm_strided_f32(
        x, output, n_cols, stride_row, stride_channel, stride_sample, eps,
        n_rows, n_channels,
        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.program_id(1)
    sample = tl.program_id(2)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    input_base = row * stride_row + channel * stride_channel + sample * stride_sample
    values = tl.load(x + input_base + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, 0.0)
    scale = 1.0 / tl.maximum(tl.sqrt(tl.sum(values * values, axis=0)), eps)
    # Keep the output layout explicit in the ABI.  Using tl.num_programs here
    # makes Triton add hidden block-count kernargs, which the portable HSACO
    # launcher deliberately rejects.  The ggml tensor dimensions are passed
    # as ordinary i32 values instead.
    output_base = ((sample * n_channels + channel) * n_rows + row) * n_cols
    tl.store(output + output_base + offsets, values * scale, mask=mask)


@triton.jit
def flagos_norm_f32(x, output, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, 0.0)
    mean = tl.sum(values, axis=0) / n_cols
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    tl.store(output + row * n_cols + offsets,
             centered / tl.sqrt(variance + eps), mask=mask)


# Inclusive prefix sum along ne[0]; the gate path needs cumulative log-decays.
@triton.jit
def flagos_cumsum_f32(x, output, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, 0.0)
    tl.store(output + row * n_cols + offsets,
             tl.cumsum(values, axis=0), mask=mask)


@triton.jit
def flagos_soft_max_f32(x, mask_ptr, output, n_cols, scale, HAS_MASK: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=float("-inf")) * scale
    if HAS_MASK:
        values += tl.load(mask_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, float("-inf"))
    values = tl.exp(values - tl.max(values, axis=0))
    tl.store(output + row * n_cols + offsets,
             values / tl.sum(values, axis=0), mask=mask)


@triton.jit
def flagos_swiglu_split_f32(gate, up, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate_values = tl.load(gate + offsets, mask=mask)
    up_values = tl.load(up + offsets, mask=mask)
    values = gate_values * tl.sigmoid(gate_values) * up_values
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flagos_set_rows_f32_f16(x, row_index, output, n_cols, n_rows, n_dst_rows, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_cols * n_rows
    rows = offsets // n_cols
    cols = offsets % n_cols
    destination_rows = tl.load(row_index + rows, mask=mask, other=0)
    # An index outside the destination view would store past the allocation, which
    # the device reports as a page fault rather than a clean error. Drop those lanes.
    mask = mask & (destination_rows >= 0) & (destination_rows < n_dst_rows)
    values = tl.load(x + rows * n_cols + cols, mask=mask, other=0.0)
    tl.store(output + destination_rows * n_cols + cols, values, mask=mask)


@triton.jit(do_not_specialize=["q_per_kv"])
def flagos_flash_attn_decode_f32_f16(
    q,
    k,
    v,
    attention_mask,
    output,
    key_length,
    q_per_kv,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_output_token,
    stride_output_head,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_head = tl.program_id(0)
    kv_head = query_head // q_per_kv
    query_rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    valid_query = query_rows == 0
    q_block = tl.load(
        q
        + query_rows[:, None] * stride_q_token
        + query_head * stride_q_head
        + dims[None, :],
        mask=valid_query[:, None],
        other=0.0,
    ).to(tl.float16)
    running_max = tl.where(valid_query, -float("inf"), 0.0)
    running_sum = tl.where(valid_query, 0.0, 1.0)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for key_start in tl.range(0, key_length, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        k_block = tl.load(
            k
            + dims[:, None]
            + kv_head * stride_k_head
            + key_offsets[None, :] * stride_k_token,
            mask=key_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q_block, k_block) * scale
        scores += tl.load(
            attention_mask + key_offsets[None, :],
            mask=key_mask[None, :],
            other=-float("inf"),
        )
        valid = valid_query[:, None] & key_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        block_max = tl.where(valid_query, block_max, 0.0)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probabilities = tl.where(valid, tl.exp(scores - new_max[:, None]), 0.0)
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= rescale[:, None]
        v_block = tl.load(
            v
            + key_offsets[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dims[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(tl.float16), v_block)
        running_sum = running_sum * rescale + block_sum
        running_max = new_max

    normalized = accumulator / running_sum[:, None]
    tl.store(
        output
        + query_rows[:, None] * stride_output_token
        + query_head * stride_output_head
        + dims[None, :],
        normalized,
        mask=valid_query[:, None],
    )


@triton.jit
def flagos_rope_neox_f32(
    x,
    positions,
    output,
    n_elements,
    ne0,
    ne1,
    ne2,
    n_dims,
    freq_base,
    freq_scale,
    BLOCK: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = n_dims // 2
    mask = pair_offsets < n_elements // 2
    rows = pair_offsets // half
    pair_dims = pair_offsets % half
    row_starts = rows * ne0
    first_offsets = row_starts + pair_dims
    second_offsets = first_offsets + half
    x0 = tl.load(x + first_offsets, mask=mask)
    x1 = tl.load(x + second_offsets, mask=mask)
    token = (rows // ne1) % ne2
    position = tl.load(positions + token, mask=mask).to(tl.float32)
    exponent = -2.0 * pair_dims.to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    cos_theta = libdevice.cos(theta)
    sin_theta = libdevice.sin(theta)
    tl.store(output + first_offsets, x0 * cos_theta - x1 * sin_theta, mask=mask)
    tl.store(output + second_offsets, x0 * sin_theta + x1 * cos_theta, mask=mask)


@triton.jit
def flagos_mrope_f32(
        x, positions, output, ne0, ne1, ne2, ne3, n_dims,
        stride_x1, stride_x2, stride_x3,
        stride_y1, stride_y2, stride_y3,
        section0, section1, section2, section3,
        freq_base, freq_scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    elements = tl.arange(0, BLOCK)
    valid = elements < ne0
    half = n_dims // 2
    i3 = row // (ne1 * ne2)
    row_in_batch = row - i3 * ne1 * ne2
    i2 = row_in_batch // ne1
    i1 = row_in_batch - i2 * ne1
    input_base = i1 * stride_x1 + i2 * stride_x2 + i3 * stride_x3
    output_base = i1 * stride_y1 + i2 * stride_y2 + i3 * stride_y3

    rotate = valid & (elements < n_dims)
    pair = elements % half
    pair_valid = rotate & (pair < half)
    x0 = tl.load(x + input_base + pair, mask=pair_valid, other=0.0).to(tl.float32)
    x1 = tl.load(x + input_base + pair + half, mask=pair_valid, other=0.0).to(tl.float32)
    copied = tl.load(x + input_base + elements, mask=valid, other=0.0).to(tl.float32)

    section_total = section0 + section1 + section2 + section3
    sector = pair % section_total
    plane = tl.where(
        sector < section0, 0,
        tl.where(sector < section0 + section1, 1,
                 tl.where(sector < section0 + section1 + section2, 2, 3)))
    position = tl.load(positions + i2 + ne2 * plane, mask=pair_valid, other=0).to(tl.float32)
    exponent = -2.0 * pair.to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    rotated = tl.where(elements < half, x0 * libdevice.cos(theta) - x1 * libdevice.sin(theta),
                       x0 * libdevice.sin(theta) + x1 * libdevice.cos(theta))
    tl.store(output + output_base + elements,
             tl.where(rotate, rotated, copied), mask=valid)


@triton.jit
def flagos_rms_norm_f32(output, x, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(output + row * n_cols + cols, normalized, mask=mask)


@triton.jit
def flagos_rms_norm_mul_f32(
    norm_output,
    mul_output,
    x,
    weight,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(norm_output + row * n_cols + cols, normalized, mask=mask)
    tl.store(mul_output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_cast_f32_f16(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.load(x + offsets, mask=mask), mask=mask)


@triton.jit
def flagos_cast_f16_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask).to(tl.float32)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flagos_dequant_q4_0_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 32)
    block_byte = block * 18
    d = tl.load(weights_f16 + block * 9).to(tl.float32)
    quant_byte = tl.load(weights_u8 + block_byte + 2 + lanes % 16)
    quant = tl.where(lanes < 16, quant_byte & 15, quant_byte >> 4).to(tl.int32) - 8
    tl.store(output + block * 32 + lanes, d * quant.to(tl.float32))


@triton.jit
def flagos_dequant_q4_1_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 32)
    block_byte = block * 20
    d = tl.load(weights_f16 + block * 10).to(tl.float32)
    m = tl.load(weights_f16 + block * 10 + 1).to(tl.float32)
    quant_byte = tl.load(weights_u8 + block_byte + 4 + lanes % 16)
    quant = tl.where(lanes < 16, quant_byte & 15, quant_byte >> 4).to(tl.float32)
    tl.store(output + block * 32 + lanes, d * quant + m)


@triton.jit
def flagos_dequant_q8_0_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 32)
    block_byte = block * 34
    d = tl.load(weights_f16 + block * 17).to(tl.float32)
    quant = tl.load(weights_u8 + block_byte + 2 + lanes).to(tl.int8).to(tl.float32)
    tl.store(output + block * 32 + lanes, d * quant)


@triton.jit
def flagos_dequant_q5_k_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 256)
    block_byte = block * 176
    block_half = block_byte // 2
    d = tl.load(weights_f16 + block_half).to(tl.float32)
    dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

    group = lanes // 32
    scale_lo_index = tl.where(group < 4, group, group + 4)
    scale_lo = tl.load(weights_u8 + block_byte + 4 + scale_lo_index)
    scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
    scale = tl.where(
        group < 4,
        scale_lo & 63,
        (scale_lo & 15) | ((scale_hi >> 6) << 4),
    ).to(tl.float32)

    min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
    min_hi = tl.load(weights_u8 + block_byte + 4 + group)
    minimum = tl.where(
        group < 4,
        min_lo & 63,
        (min_lo >> 4) | ((min_hi >> 6) << 4),
    ).to(tl.float32)

    within_64 = lanes % 64
    quant_byte = tl.load(
        weights_u8 + block_byte + 48 + (lanes // 64) * 32 + within_64 % 32
    )
    low = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4)
    high_byte = tl.load(weights_u8 + block_byte + 16 + lanes % 32)
    high = (high_byte >> (group % 8)) & 1
    quant = (low | (high << 4)).to(tl.float32)
    tl.store(output + block * 256 + lanes, d * scale * quant - dmin * minimum)


@triton.jit
def flagos_dequant_q4_k_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 256)
    block_byte = block * 144
    block_half = block_byte // 2
    d = tl.load(weights_f16 + block_half).to(tl.float32)
    dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

    group = lanes // 32
    scale_lo_index = tl.where(group < 4, group, group + 4)
    scale_lo = tl.load(weights_u8 + block_byte + 4 + scale_lo_index)
    # Clamped for the same reason as in the mul_mat kernels: tl.where evaluates
    # both branches, so a negative index here reads outside the tensor.
    scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
    scale = tl.where(
        group < 4,
        scale_lo & 63,
        (scale_lo & 15) | ((scale_hi >> 6) << 4),
    ).to(tl.float32)

    min_lo_index = tl.where(group < 4, group + 4, group + 4)
    min_lo = tl.load(weights_u8 + block_byte + 4 + min_lo_index)
    min_hi = tl.load(weights_u8 + block_byte + 4 + group)
    minimum = tl.where(
        group < 4,
        min_lo & 63,
        (min_lo >> 4) | ((min_hi >> 6) << 4),
    ).to(tl.float32)

    within_64 = lanes % 64
    quant_byte = tl.load(
        weights_u8 + block_byte + 16 + (lanes // 64) * 32 + within_64 % 32
    )
    quant = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4).to(tl.float32)
    tl.store(output + block * 256 + lanes, d * scale * quant - dmin * minimum)


@triton.jit
def flagos_dequant_q6_k_f16(weights_u8, weights_f16, output):
    block = tl.program_id(0)
    lanes = tl.arange(0, 256)
    block_byte = block * 210
    d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
    half = lanes // 128
    quadrant = (lanes % 128) // 32
    lane = lanes % 32

    ql_index = half * 64 + lane + (quadrant % 2) * 32
    ql_byte = tl.load(weights_u8 + block_byte + ql_index)
    ql = tl.where(quadrant < 2, ql_byte & 15, ql_byte >> 4)
    qh_byte = tl.load(weights_u8 + block_byte + 128 + half * 32 + lane)
    qh = (qh_byte >> (quadrant * 2)) & 3
    quant = (ql | (qh << 4)).to(tl.int32) - 32

    scale_index = half * 8 + quadrant * 2 + lane // 16
    scale_u8 = tl.load(weights_u8 + block_byte + 192 + scale_index)
    scale = scale_u8.to(tl.int8).to(tl.float32)
    tl.store(output + block * 256 + lanes, d * scale * quant.to(tl.float32))


@triton.jit
def flagos_mul_mat_q4_0_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    packed_lanes = tl.arange(0, 16)
    accumulator = tl.zeros((16,), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 18
        d = tl.load(weights_f16 + packed_block * 9).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 2 + packed_lanes)
        low = (quant_byte & 15).to(tl.int32) - 8
        high = (quant_byte >> 4).to(tl.int32) - 8
        low_activation = tl.load(x + block * 32 + packed_lanes).to(tl.float32)
        high_activation = tl.load(x + block * 32 + 16 + packed_lanes).to(tl.float32)
        accumulator += d * (low.to(tl.float32) * low_activation +
                            high.to(tl.float32) * high_activation)

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q4_0_f32_narrow(weights_u8, weights_f16, x, output, k, rows,
                                    BLOCK_M: tl.constexpr):
    """Q4_0 decode GEMV row-tile variant for wave32 targets."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_lanes = tl.arange(0, 16)
    accumulator = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row_ids[:, None] * blocks_per_row + block
        block_byte = packed_block * 18
        d = tl.load(weights_f16 + packed_block * 9).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 2 + packed_lanes[None, :])
        low = (quant_byte & 15).to(tl.int32) - 8
        high = (quant_byte >> 4).to(tl.int32) - 8
        low_activation = tl.load(x + block * 32 + packed_lanes).to(tl.float32)
        high_activation = tl.load(x + block * 32 + 16 + packed_lanes).to(tl.float32)
        accumulator += d * (low.to(tl.float32) * low_activation[None, :] +
                            high.to(tl.float32) * high_activation[None, :])

    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def flagos_mul_mat_q4_1_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    packed_lanes = tl.arange(0, 16)
    accumulator = tl.zeros((16,), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 20
        d = tl.load(weights_f16 + packed_block * 10).to(tl.float32)
        m = tl.load(weights_f16 + packed_block * 10 + 1).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 4 + packed_lanes)
        low = (quant_byte & 15).to(tl.float32)
        high = (quant_byte >> 4).to(tl.float32)
        low_activation = tl.load(x + block * 32 + packed_lanes).to(tl.float32)
        high_activation = tl.load(x + block * 32 + 16 + packed_lanes).to(tl.float32)
        accumulator += (d * low + m) * low_activation + (d * high + m) * high_activation

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q8_0_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((32,), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 34
        d = tl.load(weights_f16 + packed_block * 17).to(tl.float32)
        quant = tl.load(weights_u8 + block_byte + 2 + lanes).to(tl.int8).to(tl.float32)
        activation = tl.load(x + block * 32 + lanes).to(tl.float32)
        accumulator += d * quant * activation

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q5_k_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    packed_lanes = tl.arange(0, 128)
    chunk = packed_lanes // 32
    lane = packed_lanes % 32
    low_group = chunk * 2
    high_group = low_group + 1
    low_index = chunk * 64 + lane
    high_index = low_index + 32
    accumulator = tl.zeros((128,), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 176
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

        low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                  tl.where(low_group < 4, low_group, low_group + 4))
        low_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(low_group - 4, 0))
        low_scale = tl.where(
            low_group < 4, low_scale_byte & 63,
            (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4),
        ).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
        low_minimum = tl.where(
            low_group < 4, low_min_byte & 63,
            (low_min_byte >> 4) | ((low_min_hi >> 6) << 4),
        ).to(tl.float32)

        high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                   tl.where(high_group < 4, high_group, high_group + 4))
        high_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(high_group - 4, 0))
        high_scale = tl.where(
            high_group < 4, high_scale_byte & 63,
            (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4),
        ).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
        high_minimum = tl.where(
            high_group < 4, high_min_byte & 63,
            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4),
        ).to(tl.float32)

        quant_byte = tl.load(weights_u8 + block_byte + 48 + packed_lanes)
        qh = tl.load(weights_u8 + block_byte + 16 + lane)
        low_value = d * low_scale * ((quant_byte & 15) +
                                     ((qh >> (low_group % 8)) & 1) * 16) - dmin * low_minimum
        high_value = d * high_scale * ((quant_byte >> 4) +
                                       ((qh >> (high_group % 8)) & 1) * 16) - dmin * high_minimum
        low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
        high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
        accumulator += low_value * low_activation + high_value * high_activation

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q5_k_f32_narrow16(
        weights_u8, weights_f16, x, output, k, rows,
        BLOCK_M: tl.constexpr,
):
    """Q5_K decode GEMV using one wave32 program for a row tile."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks_per_row = k // 256
    for block in tl.range(0, blocks_per_row):
        block_byte = (row_ids[:, None] * blocks_per_row + block) * 176
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        qh = tl.load(weights_u8 + block_byte + 16 + lanes[None, :])
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                      (low_group if low_group < 4 else low_group + 4))
            low_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                   (low_group - 4 if low_group >= 4 else 0))
            low_scale = (low_scale_byte & 63 if low_group < 4 else
                         (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
            low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
            low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
            low_minimum = (low_min_byte & 63 if low_group < 4 else
                           (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
            high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                       (high_group if high_group < 4 else high_group + 4))
            high_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                    (high_group - 4 if high_group >= 4 else 0))
            high_scale = (high_scale_byte & 63 if high_group < 4 else
                          (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
            high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
            high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
            high_minimum = (high_min_byte & 63 if high_group < 4 else
                            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
            qbyte = tl.load(weights_u8 + block_byte + 48 + chunk * 32 + lanes[None, :])
            low_quant = (qbyte & 15) + ((qh >> low_group) & 1) * 16
            high_quant = (qbyte >> 4) + ((qh >> high_group) & 1) * 16
            low_value = d * low_scale * low_quant.to(tl.float32) - dmin * low_minimum
            high_value = d * high_scale * high_quant.to(tl.float32) - dmin * high_minimum
            low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
            high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
            accumulator += low_value * low_activation[None, :] + high_value * high_activation[None, :]
    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def flagos_mul_mat_q4_k_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    # Decode two weights from each packed byte together. This halves the vector
    # width, avoids reading every quant byte twice for its low/high nibble, and
    # reduces the final row reduction from 256 to 128 lanes.
    packed_lanes = tl.arange(0, 128)
    chunk = packed_lanes // 32
    lane = packed_lanes % 32
    low_index = chunk * 64 + lane
    high_index = low_index + 32
    low_group = chunk * 2
    high_group = low_group + 1
    accumulator = tl.zeros((128,), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        block_byte = (row * blocks_per_row + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

        low_scale_byte = tl.load(
            weights_u8 + block_byte + 4 + tl.where(low_group < 4, low_group, low_group + 4)
        )
        low_scale_hi = tl.load(
            weights_u8 + block_byte + 4 + tl.maximum(low_group - 4, 0)
        )
        low_scale = tl.where(
            low_group < 4,
            low_scale_byte & 63,
            (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4),
        ).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
        low_minimum = tl.where(
            low_group < 4,
            low_min_byte & 63,
            (low_min_byte >> 4) | ((low_min_hi >> 6) << 4),
        ).to(tl.float32)

        high_scale_byte = tl.load(
            weights_u8 + block_byte + 4 + tl.where(high_group < 4, high_group, high_group + 4)
        )
        high_scale_hi = tl.load(
            weights_u8 + block_byte + 4 + tl.maximum(high_group - 4, 0)
        )
        high_scale = tl.where(
            high_group < 4,
            high_scale_byte & 63,
            (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4),
        ).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
        high_minimum = tl.where(
            high_group < 4,
            high_min_byte & 63,
            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4),
        )
        high_minimum = high_minimum.to(tl.float32)

        quant_byte = tl.load(weights_u8 + block_byte + 16 + packed_lanes)
        low_value = d * low_scale * (quant_byte & 15).to(tl.float32) - dmin * low_minimum
        high_value = d * high_scale * (quant_byte >> 4).to(tl.float32) - dmin * high_minimum
        low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
        high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
        accumulator += low_value * low_activation + high_value * high_activation

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q4_k_f32_narrow(weights_u8, weights_f16, x, output, k, rows):
    """Q4_K decode GEMV tuned for wave32 targets.

    The legacy kernel exposes a 128-lane logical vector.  On AMD wave32 this
    makes the compiler carry four subchunks through the reduction and leaves
    the activation loads interleaved with quant unpacking.  This variant keeps
    one physical wave (32 lanes), explicitly walks the four Q4_K subchunks,
    and computes four output rows per program.  The provider only selects it
    for row counts divisible by four; the same ABI is retained for fallback.
    """
    pid = tl.program_id(0)
    row_ids = pid * 4 + tl.arange(0, 4)
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((4, 32), dtype=tl.float32)
    blocks_per_row = k // 256
    for block in tl.range(0, blocks_per_row):
        block_byte = (row_ids[:, None] * blocks_per_row + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                      (low_group if low_group < 4 else low_group + 4))
            low_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                   (low_group - 4 if low_group >= 4 else 0))
            low_scale = (low_scale_byte & 63 if low_group < 4 else
                         (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
            low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
            low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
            low_minimum = (low_min_byte & 63 if low_group < 4 else
                           (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
            high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                       (high_group if high_group < 4 else high_group + 4))
            high_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                    (high_group - 4 if high_group >= 4 else 0))
            high_scale = (high_scale_byte & 63 if high_group < 4 else
                          (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
            high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
            high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
            high_minimum = (high_min_byte & 63 if high_group < 4 else
                            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
            qbyte = tl.load(weights_u8 + block_byte + 16 + chunk * 32 + lanes[None, :])
            low_value = d * low_scale * (qbyte & 15).to(tl.float32) - dmin * low_minimum
            high_value = d * high_scale * (qbyte >> 4).to(tl.float32) - dmin * high_minimum
            low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
            high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
            accumulator += low_value * low_activation[None, :] + high_value * high_activation[None, :]
    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def flagos_mul_mat_q4_k_f32_narrow8(weights_u8, weights_f16, x, output, k, rows):
    """Q4_K GEMV variant for AMD wave32 targets with an eight-row tile.

    This is intentionally a separate symbol from the validated four-row
    variant.  Radeon 890M microbenchmarks show that eight rows amortize the
    quant/activation address arithmetic better for the large Qwen projection
    shapes, while the separate symbol lets the runtime keep a safe fallback
    when a package or another target does not contain this experiment.
    """
    pid = tl.program_id(0)
    row_ids = pid * 8 + tl.arange(0, 8)
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((8, 32), dtype=tl.float32)
    blocks_per_row = k // 256
    for block in tl.range(0, blocks_per_row):
        block_byte = (row_ids[:, None] * blocks_per_row + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                      (low_group if low_group < 4 else low_group + 4))
            low_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                   (low_group - 4 if low_group >= 4 else 0))
            low_scale = (low_scale_byte & 63 if low_group < 4 else
                         (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
            low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
            low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
            low_minimum = (low_min_byte & 63 if low_group < 4 else
                           (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
            high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                       (high_group if high_group < 4 else high_group + 4))
            high_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                    (high_group - 4 if high_group >= 4 else 0))
            high_scale = (high_scale_byte & 63 if high_group < 4 else
                          (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
            high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
            high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
            high_minimum = (high_min_byte & 63 if high_group < 4 else
                            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
            qbyte = tl.load(weights_u8 + block_byte + 16 + chunk * 32 + lanes[None, :])
            low_value = d * low_scale * (qbyte & 15).to(tl.float32) - dmin * low_minimum
            high_value = d * high_scale * (qbyte >> 4).to(tl.float32) - dmin * high_minimum
            low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
            high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
            accumulator += low_value * low_activation[None, :] + high_value * high_activation[None, :]
    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def flagos_mul_mat_q6_k_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
    # One Q6 high-bit byte supplies four two-bit fields. Decode all four
    # corresponding weights together so qh is read once and the reduction is
    # only 64 lanes wide instead of 256.
    packed_lanes = tl.arange(0, 64)
    half = packed_lanes // 32
    lane = packed_lanes % 32
    lane_half = lane // 16
    accumulator = tl.zeros((64,), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        block_byte = (row * blocks_per_row + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        ql_a = tl.load(weights_u8 + block_byte + half * 64 + lane)
        ql_b = tl.load(weights_u8 + block_byte + half * 64 + lane + 32)
        qh = tl.load(weights_u8 + block_byte + 128 + half * 32 + lane)

        q0 = ((ql_a & 15) | (((qh >> 0) & 3) << 4)).to(tl.int32) - 32
        q1 = ((ql_b & 15) | (((qh >> 2) & 3) << 4)).to(tl.int32) - 32
        q2 = ((ql_a >> 4) | (((qh >> 4) & 3) << 4)).to(tl.int32) - 32
        q3 = ((ql_b >> 4) | (((qh >> 6) & 3) << 4)).to(tl.int32) - 32

        scale_base = block_byte + 192 + half * 8 + lane_half
        s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
        s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
        s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
        s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)

        base = block * 256 + half * 128 + lane
        a0 = tl.load(x + base + 0).to(tl.float32)
        a1 = tl.load(x + base + 32).to(tl.float32)
        a2 = tl.load(x + base + 64).to(tl.float32)
        a3 = tl.load(x + base + 96).to(tl.float32)
        accumulator += d * (
            s0 * q0.to(tl.float32) * a0
            + s1 * q1.to(tl.float32) * a1
            + s2 * q2.to(tl.float32) * a2
            + s3 * q3.to(tl.float32) * a3
        )

    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q6_k_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    # Multi-column (prefill) counterpart of flagos_mul_mat_q6_k_f32. Same Q6_K
    # block decode; each program computes one output element.
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    if row >= rows:
        return

    # Each program dequantizes a weight row once and reuses it across a tile of
    # COLS_PER_BLOCK columns. Computing one output element per program re-read
    # the entire weight matrix for every column, pinning throughput at a flat
    # ~20 GFLOPS no matter how many columns were batched.
    # Masked lanes are not reliably suppressed on this backend (see the
    # FLAGOS_ROW_WIDTH_MULTIPLE note in ggml-flagos.cpp), so clamp the tail
    # instead of masking: surplus lanes recompute the last valid column and
    # store the same correct value to the same address, which is idempotent.
    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)

    lanes = tl.arange(0, 256)
    accumulator = tl.zeros((COLS_PER_BLOCK, 256), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        block_byte = (row * blocks_per_row + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        half = lanes // 128
        quadrant = (lanes % 128) // 32
        lane = lanes % 32

        ql_index = half * 64 + lane + (quadrant % 2) * 32
        ql_byte = tl.load(weights_u8 + block_byte + ql_index)
        ql = tl.where(quadrant < 2, ql_byte & 15, ql_byte >> 4)
        qh_byte = tl.load(weights_u8 + block_byte + 128 + half * 32 + lane)
        qh = (qh_byte >> (quadrant * 2)) & 3
        quant = (ql | (qh << 4)).to(tl.int32) - 32

        scale_index = half * 8 + quadrant * 2 + lane // 16
        scale_u8 = tl.load(weights_u8 + block_byte + 192 + scale_index)
        scale = scale_u8.to(tl.int8).to(tl.float32)
        values = d * scale * quant.to(tl.float32)
        activations = tl.load(
            x + cols[:, None] * k + block * 256 + lanes[None, :]
        ).to(tl.float32)
        accumulator += values[None, :] * activations

    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q4_k_f32_tiled(weights_u8, weights_f16, x, output, k, rows, columns,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Tiled Q4_K prefill GEMM.

    Decode 64 values at a time and feed the resulting F16 tile to tl.dot.
    The explicit tile is provider-neutral at the ABI level; only the manifest
    tile metadata and launch grid are target-specific.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = row_ids < rows
    valid_cols = col_ids < columns
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    blocks_per_row = k // 256
    for block in tl.range(0, blocks_per_row):
        block_base = (row_ids[:, None] * blocks_per_row + block) * 144
        block_half = block_base // 2
        d = tl.load(weights_f16 + block_half, mask=valid_rows[:, None], other=0.0).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1, mask=valid_rows[:, None], other=0.0).to(tl.float32)
        for sub in range(4):
            lanes = sub * 64 + tl.arange(0, 64)
            group = lanes // 32
            scale_lo_index = tl.where(group < 4, group, group + 4)
            scale_lo = tl.load(weights_u8 + block_base + 4 + scale_lo_index[None, :],
                               mask=valid_rows[:, None], other=0)
            scale_hi = tl.load(weights_u8 + block_base + 4 + tl.maximum(group - 4, 0)[None, :],
                               mask=valid_rows[:, None], other=0)
            scale = tl.where(group[None, :] < 4, scale_lo & 63,
                             (scale_lo & 15) | ((scale_hi >> 6) << 4)).to(tl.float32)
            min_lo = tl.load(weights_u8 + block_base + 4 + group[None, :] + 4,
                             mask=valid_rows[:, None], other=0)
            min_hi = tl.load(weights_u8 + block_base + 4 + group[None, :],
                             mask=valid_rows[:, None], other=0)
            minimum = tl.where(group[None, :] < 4, min_lo & 63,
                               (min_lo >> 4) | ((min_hi >> 6) << 4)).to(tl.float32)
            qbyte = tl.load(weights_u8 + block_base + 16 + (lanes[None, :] // 64) * 32 + lanes[None, :] % 32,
                            mask=valid_rows[:, None], other=0)
            quant = tl.where((lanes % 64)[None, :] < 32, qbyte & 15, qbyte >> 4).to(tl.float32)
            values = (d * scale * quant - dmin * minimum).to(tl.float16)
            act = tl.load(x + col_ids[None, :] * k + block * 256 + lanes[:, None],
                          mask=valid_cols[None, :] & (lanes[:, None] < k), other=0.0).to(tl.float16)
            acc += tl.dot(values, act)
    tl.store(output + col_ids[None, :] * rows + row_ids[:, None], acc,
             mask=valid_rows[:, None] & valid_cols[None, :])


@triton.jit
def flagos_mul_mat_q6_k_f32_tiled(weights_u8, weights_f16, x, output, k, rows, columns,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Tiled Q6_K counterpart of flagos_mul_mat_q4_k_f32_tiled."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = row_ids < rows
    valid_cols = col_ids < columns
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    blocks_per_row = k // 256
    for block in tl.range(0, blocks_per_row):
        block_base = (row_ids[:, None] * blocks_per_row + block) * 210
        d = tl.load(weights_f16 + block_base // 2 + 104,
                    mask=valid_rows[:, None], other=0.0).to(tl.float32)
        for sub in range(4):
            global_lanes = sub * 64 + tl.arange(0, 64)
            half = global_lanes // 128
            quadrant = (global_lanes % 128) // 32
            lane = global_lanes % 32
            ql_index = half * 64 + lane + (quadrant % 2) * 32
            ql_byte = tl.load(weights_u8 + block_base + ql_index[None, :],
                              mask=valid_rows[:, None], other=0)
            qh_byte = tl.load(weights_u8 + block_base + 128 + half[None, :] * 32 + lane[None, :],
                              mask=valid_rows[:, None], other=0)
            ql = tl.where(quadrant[None, :] < 2, ql_byte & 15, ql_byte >> 4)
            qh = (qh_byte >> (quadrant[None, :] * 2)) & 3
            quant = ((ql | (qh << 4)).to(tl.int32) - 32).to(tl.float32)
            scale_index = half * 8 + quadrant * 2 + lane // 16
            scale = tl.load(weights_u8 + block_base + 192 + scale_index[None, :],
                            mask=valid_rows[:, None], other=0).to(tl.int8).to(tl.float32)
            values = (d * scale * quant).to(tl.float16)
            act = tl.load(x + col_ids[None, :] * k + block * 256 + global_lanes[:, None],
                          mask=valid_cols[None, :] & (global_lanes[:, None] < k), other=0.0).to(tl.float16)
            acc += tl.dot(values, act)
    tl.store(output + col_ids[None, :] * rows + row_ids[:, None], acc,
             mask=valid_rows[:, None] & valid_cols[None, :])


@triton.jit
def flagos_mul_mat_q4_0_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= rows:
        return

    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)
    packed_lanes = tl.arange(0, 16)
    accumulator = tl.zeros((COLS_PER_BLOCK, 16), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 18
        d = tl.load(weights_f16 + packed_block * 9).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 2 + packed_lanes)
        low = (quant_byte & 15).to(tl.int32) - 8
        high = (quant_byte >> 4).to(tl.int32) - 8
        low_activation = tl.load(
            x + cols[:, None] * k + block * 32 + packed_lanes[None, :]
        ).to(tl.float32)
        high_activation = tl.load(
            x + cols[:, None] * k + block * 32 + 16 + packed_lanes[None, :]
        ).to(tl.float32)
        accumulator += d * (low[None, :].to(tl.float32) * low_activation +
                            high[None, :].to(tl.float32) * high_activation)

    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q4_1_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= rows:
        return

    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)
    packed_lanes = tl.arange(0, 16)
    accumulator = tl.zeros((COLS_PER_BLOCK, 16), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 20
        d = tl.load(weights_f16 + packed_block * 10).to(tl.float32)
        m = tl.load(weights_f16 + packed_block * 10 + 1).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 4 + packed_lanes)
        low = quant_byte & 15
        high = quant_byte >> 4
        low_activation = tl.load(
            x + cols[:, None] * k + block * 32 + packed_lanes[None, :]).to(tl.float32)
        high_activation = tl.load(
            x + cols[:, None] * k + block * 32 + 16 + packed_lanes[None, :]).to(tl.float32)
        accumulator += (d * low[None, :] + m) * low_activation
        accumulator += (d * high[None, :] + m) * high_activation

    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q8_0_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= rows:
        return

    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((COLS_PER_BLOCK, 32), dtype=tl.float32)
    blocks_per_row = k // 32

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 34
        d = tl.load(weights_f16 + packed_block * 17).to(tl.float32)
        quant = tl.load(weights_u8 + block_byte + 2 + lanes).to(tl.int8).to(tl.float32)
        activation = tl.load(
            x + cols[:, None] * k + block * 32 + lanes[None, :]).to(tl.float32)
        accumulator += d * quant[None, :] * activation

    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q5_k_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= rows:
        return

    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)
    packed_lanes = tl.arange(0, 128)
    chunk = packed_lanes // 32
    lane = packed_lanes % 32
    low_group = chunk * 2
    high_group = low_group + 1
    low_index = chunk * 64 + lane
    high_index = low_index + 32
    accumulator = tl.zeros((COLS_PER_BLOCK, 128), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        packed_block = row * blocks_per_row + block
        block_byte = packed_block * 176
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

        low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                  tl.where(low_group < 4, low_group, low_group + 4))
        low_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(low_group - 4, 0))
        low_scale = tl.where(low_group < 4, low_scale_byte & 63,
                             (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
        low_minimum = tl.where(low_group < 4, low_min_byte & 63,
                               (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
        high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                   tl.where(high_group < 4, high_group, high_group + 4))
        high_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(high_group - 4, 0))
        high_scale = tl.where(high_group < 4, high_scale_byte & 63,
                              (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
        high_minimum = tl.where(high_group < 4, high_min_byte & 63,
                                (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)

        quant_byte = tl.load(weights_u8 + block_byte + 48 + packed_lanes)
        qh = tl.load(weights_u8 + block_byte + 16 + lane)
        low_value = d * low_scale * ((quant_byte & 15) +
                                     ((qh >> (low_group % 8)) & 1) * 16) - dmin * low_minimum
        high_value = d * high_scale * ((quant_byte >> 4) +
                                       ((qh >> (high_group % 8)) & 1) * 16) - dmin * high_minimum
        low_activation = tl.load(x + cols[:, None] * k + block * 256 + low_index[None, :]).to(tl.float32)
        high_activation = tl.load(x + cols[:, None] * k + block * 256 + high_index[None, :]).to(tl.float32)
        accumulator += low_value[None, :] * low_activation + high_value[None, :] * high_activation

    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q4_k_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    # Fused dequant + GEMM for multi-column (prefill) case.
    # Each thread block computes one output element: output[row, col] = weights[row, :] @ x[:, col]
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    if row >= rows:
        return

    # Each program dequantizes a weight row once and reuses it across a tile of
    # COLS_PER_BLOCK columns. Computing one output element per program re-read
    # the entire weight matrix for every column, pinning throughput at a flat
    # ~20 GFLOPS no matter how many columns were batched.
    # Masked lanes are not reliably suppressed on this backend (see the
    # FLAGOS_ROW_WIDTH_MULTIPLE note in ggml-flagos.cpp), so clamp the tail
    # instead of masking: surplus lanes recompute the last valid column and
    # store the same correct value to the same address, which is idempotent.
    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK),
                      columns - 1)

    lanes = tl.arange(0, 256)
    accumulator = tl.zeros((COLS_PER_BLOCK, 256), dtype=tl.float32)
    blocks_per_row = k // 256

    for block in tl.range(0, blocks_per_row):
        block_byte = (row * blocks_per_row + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)

        group = lanes // 32
        scale_lo_index = tl.where(group < 4, group, group + 4)
        scale_lo = tl.load(weights_u8 + block_byte + 4 + scale_lo_index)
        # scale_hi is only used when group >= 4, but tl.where evaluates both
        # branches, so the index must stay in range: group - 4 underflows to a
        # negative offset for group < 4 and reads outside the weight tensor,
        # which faults with INVALID_ADDRESS_SPACE whenever the tensor sits near
        # the start of its buffer.
        scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
        scale = tl.where(
            group < 4,
            scale_lo & 63,
            (scale_lo & 15) | ((scale_hi >> 6) << 4),
        ).to(tl.float32)

        min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
        min_hi = tl.load(weights_u8 + block_byte + 4 + group)
        minimum = tl.where(
            group < 4,
            min_lo & 63,
            (min_lo >> 4) | ((min_hi >> 6) << 4),
        ).to(tl.float32)

        within_64 = lanes % 64
        quant_byte = tl.load(
            weights_u8 + block_byte + 16 + (lanes // 64) * 32 + within_64 % 32
        )
        quant = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4).to(tl.float32)
        values = d * scale * quant - dmin * minimum
        # ggml passes src1 as f32[k, columns] with each column contiguous along
        # k, so each column starts at its index times k.
        activations = tl.load(
            x + cols[:, None] * k + block * 256 + lanes[None, :]
        ).to(tl.float32)
        accumulator += values[None, :] * activations

    # dst is f32[rows, columns] with rows contiguous, matching src1's layout.
    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


def make_q4_0_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260902)
    packed = torch.zeros((rows, blocks, Q4_0_BLOCK_BYTES), dtype=torch.uint8)
    scales = torch.rand((rows, blocks), generator=generator, dtype=torch.float32).mul_(0.1).half()
    packed[:, :, :2] = scales.view(torch.uint8).reshape(rows, blocks, 2)
    packed[:, :, 2:] = torch.randint(
        0, 256, (rows, blocks, Q4_0_BLOCK_BYTES - 2), generator=generator, dtype=torch.uint8)

    dequantized = torch.empty((rows, blocks, QK4_0), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            for index in range(QK4_0):
                byte = int(packed[row, block, 2 + index % 16])
                quant = (byte & 15) if index < 16 else (byte >> 4)
                dequantized[row, block, index] = float(scales[row, block]) * (quant - 8)
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK4_0).cuda()


def make_q4_1_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260903)
    packed = torch.zeros((rows, blocks, Q4_1_BLOCK_BYTES), dtype=torch.uint8)
    dm = torch.randn((rows, blocks, 2), generator=generator, dtype=torch.float32).mul_(0.1).half()
    packed[:, :, :4] = dm.view(torch.uint8).reshape(rows, blocks, 4)
    packed[:, :, 4:] = torch.randint(
        0, 256, (rows, blocks, Q4_1_BLOCK_BYTES - 4), generator=generator, dtype=torch.uint8)
    dequantized = torch.empty((rows, blocks, QK4_1), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            d = float(dm[row, block, 0])
            m = float(dm[row, block, 1])
            for index in range(QK4_1):
                byte = int(packed[row, block, 4 + index % 16])
                quant = byte & 15 if index < 16 else byte >> 4
                dequantized[row, block, index] = d * quant + m
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK4_1).cuda()


def make_q8_0_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260904)
    packed = torch.zeros((rows, blocks, Q8_0_BLOCK_BYTES), dtype=torch.uint8)
    scales = torch.randn((rows, blocks), generator=generator, dtype=torch.float32).mul_(0.1).half()
    packed[:, :, :2] = scales.view(torch.uint8).reshape(rows, blocks, 2)
    signed = torch.randint(-127, 128, (rows, blocks, QK8_0), generator=generator, dtype=torch.int8)
    packed[:, :, 2:] = signed.view(torch.uint8)
    dequantized = (signed.to(torch.float32) * scales[:, :, None].to(torch.float32))
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK8_0).cuda()


def make_q4_k_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260813)
    packed = torch.zeros((rows, blocks, Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    dm = torch.rand((rows, blocks, 2), generator=generator, dtype=torch.float32).mul_(0.02).half()
    packed[:, :, :4] = dm.view(torch.uint8).reshape(rows, blocks, 4)
    packed[:, :, 4:16] = torch.randint(0, 256, (rows, blocks, 12), generator=generator, dtype=torch.uint8)
    packed[:, :, 16:] = torch.randint(0, 256, (rows, blocks, 128), generator=generator, dtype=torch.uint8)

    dequantized = torch.empty((rows, blocks, QK_K), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            scales = packed[row, block, 4:16]
            quants = packed[row, block, 16:]
            d = float(dm[row, block, 0])
            dmin = float(dm[row, block, 1])
            for index in range(QK_K):
                group = index // 32
                if group < 4:
                    scale = int(scales[group]) & 63
                    minimum = int(scales[group + 4]) & 63
                else:
                    scale = (int(scales[group + 4]) & 15) | ((int(scales[group - 4]) >> 6) << 4)
                    minimum = (int(scales[group + 4]) >> 4) | ((int(scales[group]) >> 6) << 4)
                within_64 = index % 64
                quant_byte = int(quants[(index // 64) * 32 + within_64 % 32])
                quant = quant_byte & 15 if within_64 < 32 else quant_byte >> 4
                dequantized[row, block, index] = d * scale * quant - dmin * minimum
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK_K).cuda()


def make_q5_k_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260815)
    packed = torch.zeros((rows, blocks, Q5_K_BLOCK_BYTES), dtype=torch.uint8)
    dm = torch.rand((rows, blocks, 2), generator=generator, dtype=torch.float32).mul_(0.02).half()
    packed[:, :, :4] = dm.view(torch.uint8).reshape(rows, blocks, 4)
    packed[:, :, 4:48] = torch.randint(0, 256, (rows, blocks, 44), generator=generator, dtype=torch.uint8)

    dequantized = torch.empty((rows, blocks, QK_K), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            scales = packed[row, block, 4:16]
            qh = packed[row, block, 16:48]
            quants = packed[row, block, 48:]
            d = float(dm[row, block, 0])
            dmin = float(dm[row, block, 1])
            for index in range(QK_K):
                group = index // 32
                if group < 4:
                    scale = int(scales[group]) & 63
                    minimum = int(scales[group + 4]) & 63
                else:
                    scale = (int(scales[group + 4]) & 15) | ((int(scales[group - 4]) >> 6) << 4)
                    minimum = (int(scales[group + 4]) >> 4) | ((int(scales[group]) >> 6) << 4)
                within_64 = index % 64
                quant_byte = int(quants[(index // 64) * 32 + within_64 % 32])
                quant = quant_byte & 15 if within_64 < 32 else quant_byte >> 4
                quant += ((int(qh[index % 32]) >> group) & 1) * 16
                dequantized[row, block, index] = d * scale * quant - dmin * minimum
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK_K).cuda()


def make_q6_k_weights(rows: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260814)
    packed = torch.zeros((rows, blocks, Q6_K_BLOCK_BYTES), dtype=torch.uint8)
    packed[:, :, :192] = torch.randint(0, 256, (rows, blocks, 192), generator=generator, dtype=torch.uint8)
    scales = torch.randint(-128, 128, (rows, blocks, 16), generator=generator, dtype=torch.int8)
    packed[:, :, 192:208] = scales.view(torch.uint8)
    d = torch.rand((rows, blocks), generator=generator, dtype=torch.float32).mul_(0.02).half()
    packed[:, :, 208:210] = d.view(torch.uint8).reshape(rows, blocks, 2)

    dequantized = torch.empty((rows, blocks, QK_K), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            ql = packed[row, block, :128]
            qh = packed[row, block, 128:192]
            for index in range(QK_K):
                half = index // 128
                quadrant = (index % 128) // 32
                lane = index % 32
                ql_byte = int(ql[half * 64 + lane + (quadrant % 2) * 32])
                low = ql_byte & 15 if quadrant < 2 else ql_byte >> 4
                high = (int(qh[half * 32 + lane]) >> (quadrant * 2)) & 3
                quant = (low | (high << 4)) - 32
                scale = int(scales[row, block, half * 8 + quadrant * 2 + lane // 16])
                dequantized[row, block, index] = float(d[row, block]) * scale * quant
    return packed.reshape(-1).cuda(), dequantized.reshape(rows, blocks * QK_K).cuda()


def validate_ssm_conv_silu_launch_contract() -> None:
    if (SSM_CONV_SILU_BLOCK_SIZE <= 0 or
            SSM_CONV_SILU_BLOCK_SIZE & (SSM_CONV_SILU_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_SSM_CONV_SILU_BLOCK_SIZE must be a positive power of two")
    if SSM_CONV_SILU_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError(
            "FLAGOS_SSM_CONV_SILU_NUM_WARPS must be one of 1, 2, 4, or 8")


def compile_kernels(assert_close=None) -> None:
    """Compile, launch, and numerically validate the common kernel set.

    Providers may supply an ``assert_close`` implementation when their
    floating-point contract needs a documented tolerance different from the
    PyTorch default.  The validation itself remains mandatory: callers must
    not replace it with a no-op merely to collect compiler artifacts.
    """
    validate_ssm_conv_silu_launch_contract()
    if assert_close is None:
        assert_close = torch.testing.assert_close
    n_elements = 1009
    y_elements = 127
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y_add = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y_mul = torch.randn(y_elements, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    flagos_add_f32[grid](
        x,
        y_add,
        output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(output, x + y_add)

    flagos_add_repeat_f32[grid](
        x,
        y_mul,
        output,
        n_elements,
        y_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(
        output,
        x + y_mul[torch.arange(n_elements, device="cuda") % y_elements])

    flagos_mul_f32[grid](
        x,
        y_mul,
        output,
        n_elements,
        y_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    expected = x * y_mul[torch.arange(n_elements, device="cuda") % y_elements]
    assert_close(output, expected)

    flagos_scale_f32[grid](
        x,
        output,
        0.75,
        -0.25,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(output, x * 0.75 - 0.25)

    flagos_copy_f32[grid](
        x,
        output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(output, x)

    broadcast_index = torch.arange(n_elements, device="cuda") % y_elements

    flagos_sub_f32[grid](
        x, y_mul, output, n_elements, y_elements,
        BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, x - y_mul[broadcast_index])

    # Keep the divisor away from zero so the reference stays finite.
    y_div = y_mul.abs() + 0.5
    flagos_div_f32[grid](
        x, y_div, output, n_elements, y_elements,
        BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, x / y_div[broadcast_index])

    flagos_sigmoid_f32[grid](
        x, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, torch.sigmoid(x))

    flagos_exp_f32[grid](
        x, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, torch.exp(x))

    # Include a large value to exercise the overflow-avoiding branch.
    softplus_input = torch.cat([x[:-1], torch.tensor([40.0], device="cuda")])
    flagos_softplus_f32[grid](
        softplus_input, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, torch.nn.functional.softplus(softplus_input))

    flagos_fill_f32[grid](
        output, 0.375, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    assert_close(output, torch.full_like(output, 0.375))

    # Row-wise kernels: one program per row, BLOCK covering the full row.
    row_cols = 1024
    row_count = 7
    rows = torch.randn(row_count, row_cols, device="cuda", dtype=torch.float32)
    row_output = torch.empty_like(rows)
    row_grid = (row_count,)

    sums = torch.empty(row_count, device="cuda", dtype=torch.float32)
    flagos_sum_rows_f32[row_grid](
        rows, sums, row_cols, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    assert_close(sums, rows.sum(dim=1))

    flagos_l2_norm_f32[row_grid](
        rows, row_output, row_cols, 1e-12, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    assert_close(
        row_output,
        rows / rows.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12))

    l2_heads, l2_tokens, l2_seqs, l2_cols = 3, 5, 2, 128
    l2_pitch = l2_heads * l2_cols + 17
    l2_source = torch.randn(
        l2_seqs, l2_tokens, l2_pitch, device="cuda", dtype=torch.float32)
    l2_strided_output = torch.empty(
        l2_seqs, l2_tokens, l2_heads, l2_cols,
        device="cuda", dtype=torch.float32)
    flagos_l2_norm_strided_f32[(l2_heads, l2_tokens, l2_seqs)](
        l2_source,
        l2_strided_output,
        l2_cols,
        l2_cols,
        l2_pitch,
        l2_tokens * l2_pitch,
        1e-6,
        l2_heads,
        l2_tokens,
        BLOCK=ROW_BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    l2_values = l2_source[:, :, :l2_heads * l2_cols].reshape(
        l2_seqs, l2_tokens, l2_heads, l2_cols)
    l2_expected = l2_values / l2_values.pow(2).sum(
        dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    assert_close(l2_strided_output, l2_expected)

    flagos_norm_f32[row_grid](
        rows, row_output, row_cols, 1e-5, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    centered = rows - rows.mean(dim=1, keepdim=True)
    assert_close(
        row_output, centered / (centered.pow(2).mean(dim=1, keepdim=True) + 1e-5).sqrt())

    flagos_cumsum_f32[row_grid](
        rows, row_output, row_cols, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    # The tree scan and PyTorch reference accumulate in different orders.
    assert_close(
        row_output, rows.cumsum(dim=1), rtol=2e-5, atol=2e-5)

    soft_max_mask = torch.randn(row_count, row_cols, device="cuda", dtype=torch.float32)
    flagos_soft_max_f32[row_grid](
        rows, soft_max_mask, row_output, row_cols, 0.125,
        HAS_MASK=True, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    assert_close(
        row_output, torch.softmax(rows * 0.125 + soft_max_mask, dim=1))

    # Strided copy: gather a [3, 8192] window with a 4-element row pitch,
    # mirroring the Qwen3.5 conv_state view, and compare against torch.
    sne0, sne1 = 3, 8192
    strided_src = torch.randn(sne1 * 4, device="cuda", dtype=torch.float32)
    strided_dst = torch.zeros(sne0 * sne1, device="cuda", dtype=torch.float32)
    sgrid = ((sne0 * sne1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    flagos_copy_strided_f32[sgrid](
        strided_src, strided_dst,
        sne0, sne1, 1, 1,
        sne0, sne1, 1, 1,
        1, 4, 0, 0,
        1, sne0, 0, 0,
        sne0 * sne1,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    expected_strided = strided_src.reshape(sne1, 4)[:, :sne0].reshape(-1)
    assert_close(strided_dst, expected_strided)

    concat_channels, concat_tokens, concat_state = 11, 5, 3
    concat_a_storage = torch.randn(
        concat_channels, concat_state + 2, device="cuda", dtype=torch.float32)
    concat_b_storage = torch.randn(
        concat_channels, concat_tokens, device="cuda", dtype=torch.float32)
    concat_output = torch.empty(
        concat_channels, concat_state + concat_tokens,
        device="cuda", dtype=torch.float32)
    concat_elements = concat_output.numel()
    flagos_concat_f32[(triton.cdiv(concat_elements, BLOCK_SIZE),)](
        concat_a_storage,
        concat_b_storage,
        concat_output,
        0,
        concat_state + concat_tokens, concat_channels, 1, 1,
        concat_state, concat_channels, 1, 1,
        concat_a_storage.stride(1), concat_a_storage.stride(0), 0, 0,
        concat_b_storage.stride(1), concat_b_storage.stride(0), 0, 0,
        concat_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    concat_expected = torch.cat(
        [concat_a_storage[:, :concat_state], concat_b_storage], dim=1)
    assert_close(concat_output, concat_expected)

    flagos_swiglu_split_f32[grid](
        x,
        y_add,
        output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(output, torch.nn.functional.silu(x) * y_add)

    set_rows_cols = 256
    set_rows_count = 3
    set_rows_index = torch.tensor([19, 7, 23], device="cuda", dtype=torch.int64)
    set_rows_input = torch.randn((set_rows_count, set_rows_cols), device="cuda", dtype=torch.float32)
    set_rows_output = torch.zeros((32, set_rows_cols), device="cuda", dtype=torch.float16)
    flagos_set_rows_f32_f16[(triton.cdiv(set_rows_count * set_rows_cols, BLOCK_SIZE),)](
        set_rows_input,
        set_rows_index,
        set_rows_output,
        set_rows_cols,
        set_rows_count,
        set_rows_output.shape[0],
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(set_rows_output[set_rows_index].float(), set_rows_input, rtol=5e-4, atol=5e-4)

    # Out-of-range indices must be dropped, not written past the destination.
    set_rows_guard = torch.zeros((32, set_rows_cols), device="cuda", dtype=torch.float16)
    flagos_set_rows_f32_f16[(triton.cdiv(set_rows_count * set_rows_cols, BLOCK_SIZE),)](
        set_rows_input,
        torch.tensor([5, 32, -1], device="cuda", dtype=torch.int64),
        set_rows_guard,
        set_rows_cols,
        set_rows_count,
        set_rows_guard.shape[0],
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(
        set_rows_guard[5].float(), set_rows_input[0], rtol=5e-4, atol=5e-4
    )
    assert set_rows_guard[[i for i in range(32) if i != 5]].abs().sum().item() == 0.0

    attention_key_length = 61
    for attention_q_heads, attention_kv_heads in ATTENTION_TEST_CONFIGS:
        attention_q = torch.randn(
            (1, attention_q_heads, ATTENTION_HEAD_DIM), device="cuda", dtype=torch.float32
        )
        attention_k = torch.randn(
            (attention_key_length, attention_kv_heads, ATTENTION_HEAD_DIM),
            device="cuda",
            dtype=torch.float16,
        )
        attention_v = torch.randn_like(attention_k)
        attention_mask = torch.zeros(attention_key_length, device="cuda", dtype=torch.float16)
        attention_mask[-7:] = -float("inf")
        attention_output = torch.empty_like(attention_q)
        attention_scale = ATTENTION_HEAD_DIM ** -0.5
        q_per_kv = attention_q_heads // attention_kv_heads
        flagos_flash_attn_decode_f32_f16[(attention_q_heads,)](
            attention_q,
            attention_k,
            attention_v,
            attention_mask,
            attention_output,
            attention_key_length,
            q_per_kv,
            attention_q.stride(0),
            attention_q.stride(1),
            attention_k.stride(0),
            attention_k.stride(1),
            attention_v.stride(0),
            attention_v.stride(1),
            attention_output.stride(0),
            attention_output.stride(1),
            attention_scale,
            HEAD_DIM=ATTENTION_HEAD_DIM,
            BLOCK_M=ATTENTION_BLOCK_M,
            BLOCK_N=ATTENTION_BLOCK_N,
            num_warps=NUM_WARPS,
        )
        repeated_k = attention_k.float().repeat_interleave(q_per_kv, dim=1)
        repeated_v = attention_v.float().repeat_interleave(q_per_kv, dim=1)
        reference_scores = torch.einsum(
            "thd,shd->hts", attention_q.half().float(), repeated_k
        ) * attention_scale
        reference_scores += attention_mask.float()[None, None, :]
        reference_probabilities = torch.softmax(reference_scores, dim=-1)
        attention_expected = torch.einsum(
            "hts,shd->thd", reference_probabilities.half().float(), repeated_v
        )
        assert_close(attention_output, attention_expected, rtol=8e-3, atol=8e-3)

    rope_ne0 = 128
    rope_ne1 = 12
    rope_ne2 = 5
    rope_n_elements = rope_ne0 * rope_ne1 * rope_ne2
    rope_input = torch.randn(rope_n_elements, device="cuda", dtype=torch.float32)
    rope_output = torch.empty_like(rope_input)
    rope_positions = torch.randint(0, 512, (rope_ne2,), device="cuda", dtype=torch.int32)
    rope_base = 1_000_000.0
    rope_scale = 1.0
    flagos_rope_neox_f32[(triton.cdiv(rope_n_elements, BLOCK_SIZE),)](
        rope_input,
        rope_positions,
        rope_output,
        rope_n_elements,
        rope_ne0,
        rope_ne1,
        rope_ne2,
        rope_ne0,
        rope_base,
        rope_scale,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    rope_view = rope_input.reshape(rope_ne2, rope_ne1, rope_ne0)
    rope_half = rope_ne0 // 2
    rope_dims = torch.arange(rope_half, device="cuda", dtype=torch.float32)
    rope_theta = rope_positions.float()[:, None, None] * torch.exp(
        (-2.0 * rope_dims / rope_ne0) * torch.log(torch.tensor(rope_base, device="cuda"))
    )[None, None, :]
    rope_expected = torch.empty_like(rope_view)
    rope_expected[:, :, :rope_half] = (
        rope_view[:, :, :rope_half] * torch.cos(rope_theta)
        - rope_view[:, :, rope_half:] * torch.sin(rope_theta)
    )
    rope_expected[:, :, rope_half:] = (
        rope_view[:, :, :rope_half] * torch.sin(rope_theta)
        + rope_view[:, :, rope_half:] * torch.cos(rope_theta)
    )
    # Tolerance covers fp32 rounding in the sin/cos path; observed worst case on
    # KS20 is ~2.2e-4 absolute on a single element out of 7680.
    assert_close(rope_output, rope_expected.reshape(-1), rtol=1e-4, atol=5e-4)

    mrope_ne0, mrope_ne1, mrope_ne2, mrope_ne3 = 80, 3, 5, 2
    mrope_n_dims = 64
    mrope_sections = (8, 8, 8, 8)
    mrope_input = torch.randn(
        mrope_ne3, mrope_ne2, mrope_ne1, mrope_ne0,
        device="cuda", dtype=torch.float32)
    mrope_output = torch.empty_like(mrope_input)
    mrope_positions = torch.randint(
        0, 512, (4, mrope_ne2), device="cuda", dtype=torch.int32)
    flagos_mrope_f32[(mrope_ne1 * mrope_ne2 * mrope_ne3,)](
        mrope_input,
        mrope_positions,
        mrope_output,
        mrope_ne0, mrope_ne1, mrope_ne2, mrope_ne3, mrope_n_dims,
        mrope_input.stride(2), mrope_input.stride(1), mrope_input.stride(0),
        mrope_output.stride(2), mrope_output.stride(1), mrope_output.stride(0),
        *mrope_sections,
        rope_base, rope_scale,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    mrope_expected = mrope_input.clone()
    mrope_half = mrope_n_dims // 2
    mrope_pair = torch.arange(mrope_half, device="cuda")
    mrope_sector = mrope_pair % sum(mrope_sections)
    mrope_plane = torch.where(
        mrope_sector < mrope_sections[0], 0,
        torch.where(
            mrope_sector < sum(mrope_sections[:2]), 1,
            torch.where(mrope_sector < sum(mrope_sections[:3]), 2, 3)))
    mrope_freq = torch.exp(
        (-2.0 * mrope_pair / mrope_n_dims) *
        torch.log(torch.tensor(rope_base, device="cuda")))
    for token in range(mrope_ne2):
        mrope_theta = mrope_positions[mrope_plane, token].float() * mrope_freq
        first = mrope_input[:, token, :, :mrope_half]
        second = mrope_input[:, token, :, mrope_half:mrope_n_dims]
        mrope_expected[:, token, :, :mrope_half] = (
            first * torch.cos(mrope_theta) - second * torch.sin(mrope_theta))
        mrope_expected[:, token, :, mrope_half:mrope_n_dims] = (
            first * torch.sin(mrope_theta) + second * torch.cos(mrope_theta))
    assert_close(mrope_output, mrope_expected, rtol=1e-4, atol=5e-4)

    rows = 5
    n_cols = 1536
    eps = 1e-6
    norm_input = torch.randn((rows, n_cols), device="cuda", dtype=torch.float32)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    norm_output = torch.empty_like(norm_input)
    mul_output = torch.empty_like(norm_input)
    flagos_rms_norm_f32[(rows,)](
        norm_output,
        norm_input,
        n_cols,
        eps,
        BLOCK=RMS_NORM_BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    expected_norm = norm_input * torch.rsqrt(torch.mean(norm_input * norm_input, dim=1, keepdim=True) + eps)
    assert_close(norm_output, expected_norm, rtol=2e-5, atol=2e-5)

    flagos_rms_norm_mul_f32[(rows,)](
        norm_output,
        mul_output,
        norm_input,
        weight,
        n_cols,
        eps,
        BLOCK=RMS_NORM_BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(norm_output, expected_norm, rtol=2e-5, atol=2e-5)
    assert_close(mul_output, expected_norm * weight, rtol=2e-5, atol=2e-5)

    # GET_ROWS over a Q4_K table.
    rows_table = 12
    rows_blocks = 3
    rows_cols = rows_blocks * QK_K
    rows_q40_blocks = rows_cols // QK4_0
    table_q40_packed, table_q40_dequantized = make_q4_0_weights(
        rows_table, rows_q40_blocks)
    row_ids = torch.tensor([7, 0, 11, 3], device="cuda", dtype=torch.int32)
    get_rows_output = torch.empty(
        (row_ids.numel(), rows_cols), device="cuda", dtype=torch.float32
    )
    flagos_get_rows_q4_0_f32[(row_ids.numel(), rows_q40_blocks)](
        table_q40_packed,
        table_q40_packed.view(torch.float16),
        row_ids,
        get_rows_output,
        rows_cols,
        num_warps=NUM_WARPS,
    )
    assert_close(
        get_rows_output, table_q40_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3
    )

    table_q41_packed, table_q41_dequantized = make_q4_1_weights(rows_table, rows_q40_blocks)
    flagos_get_rows_q4_1_f32[(row_ids.numel(), rows_q40_blocks)](
        table_q41_packed, table_q41_packed.view(torch.float16), row_ids,
        get_rows_output, rows_cols, num_warps=NUM_WARPS)
    assert_close(
        get_rows_output, table_q41_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3)

    table_q80_packed, table_q80_dequantized = make_q8_0_weights(rows_table, rows_q40_blocks)
    flagos_get_rows_q8_0_f32[(row_ids.numel(), rows_q40_blocks)](
        table_q80_packed, table_q80_packed.view(torch.float16), row_ids,
        get_rows_output, rows_cols, num_warps=NUM_WARPS)
    assert_close(
        get_rows_output, table_q80_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3)

    table_packed, table_dequantized = make_q4_k_weights(rows_table, rows_blocks)
    flagos_get_rows_q4_k_f32[(row_ids.numel(), rows_blocks)](
        table_packed,
        table_packed.view(torch.float16),
        row_ids,
        get_rows_output,
        rows_cols,
        num_warps=NUM_WARPS,
    )
    assert_close(
        get_rows_output, table_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3
    )

    table_q5_packed, table_q5_dequantized = make_q5_k_weights(rows_table, rows_blocks)
    flagos_get_rows_q5_k_f32[(row_ids.numel(), rows_blocks)](
        table_q5_packed,
        table_q5_packed.view(torch.float16),
        row_ids,
        get_rows_output,
        rows_cols,
        num_warps=NUM_WARPS,
    )
    assert_close(
        get_rows_output, table_q5_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3
    )

    # Qwen3-4B uses a Q6_K token embedding table in Q4_K_M files.
    table_q6_packed, table_q6_dequantized = make_q6_k_weights(rows_table, rows_blocks)
    flagos_get_rows_q6_k_f32[(row_ids.numel(), rows_blocks)](
        table_q6_packed,
        table_q6_packed.view(torch.float16),
        row_ids,
        get_rows_output,
        rows_cols,
        num_warps=NUM_WARPS,
    )
    assert_close(
        get_rows_output, table_q6_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3
    )

    dense_cols = 769
    dense_table = torch.randn((rows_table, dense_cols), device="cuda", dtype=torch.float32)
    dense_output = torch.empty(
        (row_ids.numel(), dense_cols), device="cuda", dtype=torch.float32
    )
    flagos_get_rows_f32[(row_ids.numel(), triton.cdiv(dense_cols, BLOCK_SIZE))](
        dense_table,
        row_ids,
        dense_output,
        dense_cols,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(dense_output, dense_table[row_ids.long()])

    # SSM_CONV: causal depthwise conv1d over the rolling state window.
    conv_d_conv = 4
    conv_d_inner = 96
    conv_tokens = 5
    conv_seqs = 2
    conv_ncs = conv_d_conv - 1 + conv_tokens
    conv_state = torch.randn(
        (conv_seqs, conv_d_inner, conv_ncs), device="cuda", dtype=torch.float32
    )
    conv_weight = torch.randn(
        (conv_d_inner, conv_d_conv), device="cuda", dtype=torch.float32
    )
    conv_output = torch.empty(
        (conv_seqs, conv_tokens, conv_d_inner), device="cuda", dtype=torch.float32
    )
    flagos_ssm_conv_f32[(conv_tokens, conv_seqs, triton.cdiv(conv_d_inner, BLOCK_SIZE))](
        conv_state,
        conv_weight,
        conv_output,
        conv_d_conv,
        conv_d_inner,
        conv_tokens,
        conv_seqs,
        conv_state.stride(1),
        conv_state.stride(0),
        conv_weight.stride(0),
        conv_output.stride(1),
        conv_output.stride(0),
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    # Reference: for each token the window is state[:, :, token:token+d_conv].
    conv_expected = torch.empty_like(conv_output)
    for conv_t in range(conv_tokens):
        window = conv_state[:, :, conv_t:conv_t + conv_d_conv]
        conv_expected[:, conv_t, :] = (window * conv_weight[None, :, :]).sum(dim=-1)
    assert_close(conv_output, conv_expected, rtol=2e-5, atol=2e-5)

    conv_silu_output = torch.empty_like(conv_output)
    flagos_ssm_conv_silu_f32[(
            conv_tokens, conv_seqs,
            triton.cdiv(conv_d_inner, SSM_CONV_SILU_BLOCK_SIZE))](
        conv_state,
        conv_weight,
        conv_silu_output,
        conv_d_conv,
        conv_d_inner,
        conv_tokens,
        conv_seqs,
        conv_state.stride(1),
        conv_state.stride(0),
        conv_weight.stride(0),
        conv_silu_output.stride(1),
        conv_silu_output.stride(0),
        BLOCK=SSM_CONV_SILU_BLOCK_SIZE,
        num_warps=SSM_CONV_SILU_NUM_WARPS,
    )
    assert_close(
        conv_silu_output, torch.nn.functional.silu(conv_expected),
        rtol=2e-5, atol=2e-5)

    gdn_state_size = 128
    gdn_q_heads, gdn_heads = 2, 4
    gdn_tokens, gdn_seqs, gdn_q_seqs = 4, 2, 1
    gdn_snapshots = 3
    gdn_q = torch.randn(
        gdn_q_seqs, gdn_tokens, gdn_q_heads, gdn_state_size,
        device="cuda", dtype=torch.float32)
    gdn_k = torch.randn_like(gdn_q)
    gdn_v = torch.randn(
        gdn_seqs, gdn_tokens, gdn_heads, gdn_state_size,
        device="cuda", dtype=torch.float32)
    gdn_gate = torch.randn(
        gdn_seqs, gdn_tokens, gdn_heads, 1,
        device="cuda", dtype=torch.float32).mul_(0.02)
    gdn_beta = torch.sigmoid(torch.randn_like(gdn_gate))
    gdn_state = torch.randn(
        gdn_seqs, gdn_heads, gdn_state_size, gdn_state_size,
        device="cuda", dtype=torch.float32).mul_(0.02)
    gdn_attention_elements = gdn_state_size * gdn_heads * gdn_tokens * gdn_seqs
    gdn_snapshot_stride = gdn_state_size * gdn_state_size * gdn_heads * gdn_seqs
    gdn_output = torch.empty(
        gdn_attention_elements + gdn_snapshots * gdn_snapshot_stride,
        device="cuda", dtype=torch.float32)
    gdn_scale = gdn_state_size ** -0.5
    flagos_gated_delta_net_scalar_f32[(
        gdn_heads, gdn_seqs, triton.cdiv(gdn_state_size, 4))](
        gdn_q, gdn_k, gdn_v, gdn_gate, gdn_beta, gdn_state, gdn_output,
        gdn_state_size, gdn_heads, gdn_tokens, gdn_seqs,
        gdn_q.stride(2), gdn_q.stride(1), gdn_q.stride(0),
        gdn_v.stride(2), gdn_v.stride(1), gdn_v.stride(0),
        gdn_beta.stride(2), gdn_beta.stride(1), gdn_beta.stride(0),
        gdn_q_heads, gdn_seqs // gdn_q_seqs, gdn_snapshots, gdn_scale,
        BLOCK=gdn_state_size,
        COLS=4,
        num_warps=4,
    )
    gdn_fused_output = torch.empty_like(gdn_output)
    gdn_cache_output = torch.empty(
        gdn_snapshots, gdn_seqs, gdn_heads, gdn_state_size, gdn_state_size,
        device="cuda", dtype=torch.float32)
    flagos_gated_delta_net_scalar_f32_cache[(
        gdn_heads, gdn_seqs, triton.cdiv(gdn_state_size, 4))](
        gdn_q, gdn_k, gdn_v, gdn_gate, gdn_beta, gdn_state,
        gdn_fused_output, gdn_cache_output,
        gdn_state_size, gdn_heads, gdn_tokens, gdn_seqs,
        gdn_q.stride(2), gdn_q.stride(1), gdn_q.stride(0),
        gdn_v.stride(2), gdn_v.stride(1), gdn_v.stride(0),
        gdn_beta.stride(2), gdn_beta.stride(1), gdn_beta.stride(0),
        gdn_q_heads, gdn_seqs // gdn_q_seqs, gdn_snapshots, gdn_scale,
        gdn_cache_output.stride(0),
        BLOCK=gdn_state_size,
        COLS=4,
        num_warps=4,
    )
    gdn_expected_attention = torch.empty_like(gdn_v)
    gdn_expected_snapshots = torch.empty(
        gdn_snapshots, gdn_seqs, gdn_heads, gdn_state_size, gdn_state_size,
        device="cuda", dtype=torch.float32)
    for sequence in range(gdn_seqs):
        for head in range(gdn_heads):
            state_ref = gdn_state[sequence, head].clone()
            for token in range(gdn_tokens):
                q_ref = gdn_q[sequence // (gdn_seqs // gdn_q_seqs), token,
                              head % gdn_q_heads]
                k_ref = gdn_k[sequence // (gdn_seqs // gdn_q_seqs), token,
                              head % gdn_q_heads]
                state_ref *= torch.exp(gdn_gate[sequence, token, head, 0])
                delta = (gdn_v[sequence, token, head] - state_ref @ k_ref) * \
                    gdn_beta[sequence, token, head, 0]
                state_ref += delta[:, None] * k_ref[None, :]
                gdn_expected_attention[sequence, token, head] = state_ref @ q_ref * gdn_scale
                snapshot = gdn_tokens - 1 - token
                if snapshot < gdn_snapshots:
                    gdn_expected_snapshots[snapshot, sequence, head] = state_ref
    assert_close(
        gdn_output[:gdn_attention_elements].view_as(gdn_expected_attention),
        gdn_expected_attention, rtol=2e-4, atol=2e-4)
    assert_close(
        gdn_output[gdn_attention_elements:].view_as(gdn_expected_snapshots),
        gdn_expected_snapshots, rtol=2e-4, atol=2e-4)
    assert_close(gdn_fused_output, gdn_output, rtol=2e-4, atol=2e-4)
    assert_close(gdn_cache_output, gdn_expected_snapshots, rtol=2e-4, atol=2e-4)

    cast_output = torch.empty(n_elements, device="cuda", dtype=torch.float16)
    flagos_cast_f32_f16[grid](
        x,
        cast_output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(cast_output.float(), x.half().float())

    cast_back_output = torch.empty(n_elements, device="cuda", dtype=torch.float32)
    flagos_cast_f16_f32[grid](
        cast_output,
        cast_back_output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    assert_close(cast_back_output, x.half().float())

    mat_rows = 17
    mat_blocks = 6
    mat_k = mat_blocks * QK_K
    activation = torch.randn(mat_k, device="cuda", dtype=torch.float32)
    mat_output = torch.empty(mat_rows, device="cuda", dtype=torch.float32)

    q40_blocks = mat_k // QK4_0
    q40_packed, q40_dequantized = make_q4_0_weights(mat_rows, q40_blocks)
    q40_dequantized_f16 = torch.empty_like(q40_dequantized, dtype=torch.float16)
    flagos_dequant_q4_0_f16[(mat_rows * q40_blocks,)](
        q40_packed,
        q40_packed.view(torch.float16),
        q40_dequantized_f16,
        num_warps=NUM_WARPS,
    )
    assert_close(q40_dequantized_f16.float(), q40_dequantized.half().float())
    flagos_mul_mat_q4_0_f32[(mat_rows,)](
        q40_packed,
        q40_packed.view(torch.float16),
        activation,
        mat_output,
        mat_k,
        mat_rows,
        num_warps=GEMV_NUM_WARPS,
    )
    assert_close(mat_output, q40_dequantized @ activation, rtol=2e-5, atol=2e-4)

    if os.environ.get("FLAGOS_Q40_GEMV_NARROW_ENABLE", "0") == "1":
        narrow_block_m = int(os.environ.get("FLAGOS_Q40_GEMV_NARROW_BLOCK_M", "8"))
        if (narrow_block_m <= 1 or narrow_block_m > 32 or
                narrow_block_m & (narrow_block_m - 1)):
            raise ValueError("FLAGOS_Q40_GEMV_NARROW_BLOCK_M must be a power of two from 2 to 32")
        narrow_rows = 32
        narrow_packed, narrow_dequantized = make_q4_0_weights(narrow_rows, q40_blocks)
        narrow_output = torch.empty(narrow_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q4_0_f32_narrow[(narrow_rows // narrow_block_m,)](
            narrow_packed,
            narrow_packed.view(torch.float16),
            activation,
            narrow_output,
            mat_k,
            narrow_rows,
            BLOCK_M=narrow_block_m,
            num_warps=1,
            waves_per_eu=4,
        )
        assert_close(
            narrow_output, narrow_dequantized @ activation,
            rtol=2e-5, atol=2e-4,
        )

    q41_packed, q41_dequantized = make_q4_1_weights(mat_rows, q40_blocks)
    q41_dequantized_f16 = torch.empty_like(q41_dequantized, dtype=torch.float16)
    flagos_dequant_q4_1_f16[(mat_rows * q40_blocks,)](
        q41_packed, q41_packed.view(torch.float16), q41_dequantized_f16,
        num_warps=NUM_WARPS)
    assert_close(q41_dequantized_f16.float(), q41_dequantized.half().float())
    flagos_mul_mat_q4_1_f32[(mat_rows,)](
        q41_packed, q41_packed.view(torch.float16), activation, mat_output,
        mat_k, mat_rows, num_warps=GEMV_NUM_WARPS)
    assert_close(mat_output, q41_dequantized @ activation, rtol=2e-5, atol=2e-4)

    q80_packed, q80_dequantized = make_q8_0_weights(mat_rows, q40_blocks)
    q80_dequantized_f16 = torch.empty_like(q80_dequantized, dtype=torch.float16)
    flagos_dequant_q8_0_f16[(mat_rows * q40_blocks,)](
        q80_packed, q80_packed.view(torch.float16), q80_dequantized_f16,
        num_warps=NUM_WARPS)
    assert_close(q80_dequantized_f16.float(), q80_dequantized.half().float())
    flagos_mul_mat_q8_0_f32[(mat_rows,)](
        q80_packed, q80_packed.view(torch.float16), activation, mat_output,
        mat_k, mat_rows, num_warps=GEMV_NUM_WARPS)
    assert_close(mat_output, q80_dequantized @ activation, rtol=2e-5, atol=2e-4)

    q4_packed, q4_dequantized = make_q4_k_weights(mat_rows, mat_blocks)
    q4_dequantized_f16 = torch.empty_like(q4_dequantized, dtype=torch.float16)
    flagos_dequant_q4_k_f16[(mat_rows * mat_blocks,)](
        q4_packed,
        q4_packed.view(torch.float16),
        q4_dequantized_f16,
        num_warps=NUM_WARPS,
    )
    assert_close(q4_dequantized_f16.float(), q4_dequantized.half().float())
    flagos_mul_mat_q4_k_f32[(mat_rows,)](
        q4_packed,
        q4_packed.view(torch.float16),
        activation,
        mat_output,
        mat_k,
        mat_rows,
        num_warps=GEMV_NUM_WARPS,
    )
    assert_close(mat_output, q4_dequantized @ activation, rtol=2e-5, atol=2e-4)

    q5_packed, q5_dequantized = make_q5_k_weights(mat_rows, mat_blocks)
    q5_dequantized_f16 = torch.empty_like(q5_dequantized, dtype=torch.float16)
    flagos_dequant_q5_k_f16[(mat_rows * mat_blocks,)](
        q5_packed,
        q5_packed.view(torch.float16),
        q5_dequantized_f16,
        num_warps=NUM_WARPS,
    )
    assert_close(q5_dequantized_f16.float(), q5_dequantized.half().float(), rtol=2e-3, atol=2e-3)
    flagos_mul_mat_q5_k_f32[(mat_rows,)](
        q5_packed,
        q5_packed.view(torch.float16),
        activation,
        mat_output,
        mat_k,
        mat_rows,
        num_warps=GEMV_NUM_WARPS,
    )
    assert_close(mat_output, q5_dequantized @ activation, rtol=2e-5, atol=2e-4)

    if os.environ.get("FLAGOS_Q5_GEMV_NARROW16_ENABLE", "0") == "1":
        q5_narrow_rows = 16
        q5_narrow_output = torch.empty(
            q5_narrow_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q5_k_f32_narrow16[(1,)](
            q5_packed,
            q5_packed.view(torch.float16),
            activation,
            q5_narrow_output,
            mat_k,
            q5_narrow_rows,
            BLOCK_M=16,
            num_warps=1,
        )
        assert_close(
            q5_narrow_output, q5_dequantized[:q5_narrow_rows] @ activation,
            rtol=2e-5, atol=2e-3,
        )

    # These wave32 GEMV experiments are AMD-specific.  Keep them out of the
    # shared Denglin/CUDA package unless the AMD generator explicitly opts in.
    if os.environ.get("FLAGOS_Q4_GEMV_NARROW_ENABLE", "0") == "1":
        narrow_rows = 16
        narrow_output = torch.empty(narrow_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q4_k_f32_narrow[(narrow_rows // 4,)](
            q4_packed,
            q4_packed.view(torch.float16),
            activation,
            narrow_output,
            mat_k,
            narrow_rows,
            num_warps=1,
        )
        assert_close(
            narrow_output, q4_dequantized[:narrow_rows] @ activation,
            rtol=2e-5, atol=2e-3,
        )

    if os.environ.get("FLAGOS_Q4_GEMV_NARROW8_ENABLE", "0") == "1":
        narrow8_rows = 16
        narrow8_output = torch.empty(narrow8_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q4_k_f32_narrow8[(narrow8_rows // 8,)](
            q4_packed,
            q4_packed.view(torch.float16),
            activation,
            narrow8_output,
            mat_k,
            narrow8_rows,
            num_warps=1,
        )
        assert_close(
            narrow8_output, q4_dequantized[:narrow8_rows] @ activation,
            rtol=2e-5, atol=2e-3,
        )

    # Test batched (multi-column) GEMM for prefill
    # Deliberately not a multiple of MUL_MAT_COLS_PER_BLOCK so the tail path
    # in the batched GEMMs is exercised; 32 columns hid a tail bug entirely.
    mat_columns = 37
    # ggml layout: each of the `mat_columns` activation columns is contiguous
    # along k, so store them as rows here and transpose in the reference.
    activation_batched = torch.randn(mat_columns, mat_k, device="cuda", dtype=torch.float32)
    mat_output_batched = torch.zeros(mat_columns * mat_rows, device="cuda", dtype=torch.float32)
    flagos_mul_mat_q4_0_f32_batched[(
        mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q40_packed,
        q40_packed.view(torch.float16),
        activation_batched,
        mat_output_batched,
        mat_k,
        mat_rows,
        mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK,
        num_warps=NUM_WARPS,
    )
    assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q40_dequantized @ activation_batched.T,
        rtol=2e-5,
        atol=2e-4,
    )
    mat_output_batched.zero_()
    flagos_mul_mat_q4_1_f32_batched[(mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q41_packed, q41_packed.view(torch.float16), activation_batched,
        mat_output_batched, mat_k, mat_rows, mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK, num_warps=NUM_WARPS)
    assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q41_dequantized @ activation_batched.T, rtol=2e-5, atol=2e-4)
    mat_output_batched.zero_()
    flagos_mul_mat_q8_0_f32_batched[(mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q80_packed, q80_packed.view(torch.float16), activation_batched,
        mat_output_batched, mat_k, mat_rows, mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK, num_warps=NUM_WARPS)
    assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q80_dequantized @ activation_batched.T, rtol=2e-5, atol=2e-4)
    mat_output_batched.zero_()
    flagos_mul_mat_q4_k_f32_batched[(mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q4_packed,
        q4_packed.view(torch.float16),
        activation_batched,
        mat_output_batched,
        mat_k,
        mat_rows,
        mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK,
        num_warps=NUM_WARPS,
    )
    assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q4_dequantized @ activation_batched.T,
        rtol=2e-5,
        atol=2e-4,
    )

    mat_output_batched.zero_()
    flagos_mul_mat_q5_k_f32_batched[(mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q5_packed,
        q5_packed.view(torch.float16),
        activation_batched,
        mat_output_batched,
        mat_k,
        mat_rows,
        mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK,
        num_warps=NUM_WARPS,
    )
    assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q5_dequantized @ activation_batched.T,
        rtol=2e-5,
        atol=2e-4,
    )

    q6_packed, q6_dequantized = make_q6_k_weights(mat_rows, mat_blocks)
    q6_dequantized_f16 = torch.empty_like(q6_dequantized, dtype=torch.float16)
    flagos_dequant_q6_k_f16[(mat_rows * mat_blocks,)](
        q6_packed,
        q6_packed.view(torch.float16),
        q6_dequantized_f16,
        num_warps=NUM_WARPS,
    )
    assert_close(q6_dequantized_f16.float(), q6_dequantized.half().float())

    # Batched Q6_K, checked against ggml's layout: src1 is f32[k, columns] with
    # each column contiguous, and dst is f32[rows, columns] likewise.
    q6_activation_cols = torch.randn(mat_columns, mat_k, device="cuda", dtype=torch.float32)
    q6_out_batched = torch.zeros(mat_columns * mat_rows, device="cuda", dtype=torch.float32)
    flagos_mul_mat_q6_k_f32_batched[(mat_rows, triton.cdiv(mat_columns, MUL_MAT_COLS_PER_BLOCK))](
        q6_packed,
        q6_packed.view(torch.float16),
        q6_activation_cols,
        q6_out_batched,
        mat_k,
        mat_rows,
        mat_columns,
        COLS_PER_BLOCK=MUL_MAT_COLS_PER_BLOCK,
        num_warps=NUM_WARPS,
    )
    assert_close(
        q6_out_batched.view(mat_columns, mat_rows).T,
        q6_dequantized @ q6_activation_cols.T,
        rtol=2e-5,
        atol=2e-4,
    )
    flagos_mul_mat_q6_k_f32[(mat_rows,)](
        q6_packed,
        q6_packed.view(torch.float16),
        activation,
        mat_output,
        mat_k,
        mat_rows,
        num_warps=GEMV_NUM_WARPS,
    )
    assert_close(mat_output, q6_dequantized @ activation, rtol=2e-5, atol=2e-3)

    if os.environ.get("FLAGOS_QUANT_TILE_ENABLE", "0") == "1":
        # MFMA-friendly tiled prefill variants.  Keep the dimensions small
        # enough for the AMD conformance generator while exercising tails.
        tiled_rows, tiled_columns = mat_rows, 35
        tiled_out = torch.zeros(tiled_columns * tiled_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q4_k_f32_tiled[(triton.cdiv(tiled_rows, QUANT_TILE_BLOCK_M),
                                       triton.cdiv(tiled_columns, QUANT_TILE_BLOCK_N))](
            q4_packed, q4_packed.view(torch.float16), activation_batched, tiled_out,
            mat_k, tiled_rows, tiled_columns,
            BLOCK_M=QUANT_TILE_BLOCK_M, BLOCK_N=QUANT_TILE_BLOCK_N,
            num_warps=QUANT_TILE_NUM_WARPS,
        )
        assert_close(
            tiled_out.view(tiled_columns, tiled_rows).T,
            q4_dequantized[:tiled_rows] @ activation_batched[:tiled_columns].T,
            rtol=3e-2, atol=3e-2,
        )
        tiled_q6_out = torch.zeros(tiled_columns * tiled_rows, device="cuda", dtype=torch.float32)
        flagos_mul_mat_q6_k_f32_tiled[(triton.cdiv(tiled_rows, QUANT_TILE_BLOCK_M),
                                       triton.cdiv(tiled_columns, QUANT_TILE_BLOCK_N))](
            q6_packed, q6_packed.view(torch.float16), q6_activation_cols, tiled_q6_out,
            mat_k, tiled_rows, tiled_columns,
            BLOCK_M=QUANT_TILE_BLOCK_M, BLOCK_N=QUANT_TILE_BLOCK_N,
            num_warps=QUANT_TILE_NUM_WARPS,
        )
        assert_close(
            tiled_q6_out.view(tiled_columns, tiled_rows).T,
            q6_dequantized[:tiled_rows] @ q6_activation_cols[:tiled_columns].T,
            rtol=3e-2, atol=3e-2,
        )
    torch.cuda.synchronize()


def copy_artifact(
    cache_dir: Path,
    output_dir: Path,
    kernel_name: str,
    block_size: int,
) -> dict:
    matches = list(cache_dir.rglob(f"{kernel_name}.cubin"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {kernel_name}.cubin, found {len(matches)}")

    source = matches[0]
    destination = output_dir / f"{kernel_name}.cubin"
    shutil.copyfile(source, destination)

    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    return {
        "file": destination.name,
        "name": metadata["name"],
        "shared": metadata["shared"],
        "num_warps": metadata["num_warps"],
        "warp_size": metadata["target"]["warp_size"],
        "block_size": block_size,
    }


def find_llir(cache_dir: Path, kernel_name: str) -> Path:
    matches = list(cache_dir.rglob(f"{kernel_name}.llir"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {kernel_name}.llir, found {len(matches)}")
    return matches[0]


def build_merged_module(
    cache_dir: Path,
    output_dir: Path,
    kernel_names: list[str],
    sdk_root: Path,
) -> None:
    """Merge every kernel's LLVM IR and compile it into one loadable module.

    The SDK linkers cannot do this: cuLinkAddData only accepts PTX, dllink
    rejects finished cubins, and llvm-link silently keeps just the first
    module. Merging the textual IR before dlcc is the one path that works.
    """
    merged_ir = output_dir / "flagos_kernels.ll"
    inputs = [find_llir(cache_dir, name) for name in kernel_names]
    merged = merge_llvm_modules(inputs, merged_ir, DLCC_TARGET)

    missing = [name for name in kernel_names if name not in merged]
    if missing:
        raise RuntimeError(f"merged module lost kernels: {missing}")

    llvm_as = sdk_root / "bin" / "llvm-as"
    if llvm_as.exists():
        subprocess.run([str(llvm_as), str(merged_ir), "-o", "/dev/null"], check=True)

    output = output_dir / MERGED_MODULE_NAME
    command = [str(sdk_root / "bin" / "dlcc"), *DLCC_OPTIONS,
               f"--target={DLCC_TARGET}", str(merged_ir), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dlcc failed for the merged module:\n{result.stderr}")

    merged_ir.unlink()
    print(f"merged {len(kernel_names)} kernels into {output.name} "
          f"({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path,
                        default=Path(os.environ.get("FLAGOS_SDK_ROOT", "/usr/local/dlgpu/sdk")))
    parser.add_argument("--no-merge", action="store_true",
                        help="skip the merged module and emit only per-kernel cubins")
    args = parser.parse_args()

    cache_dir_env = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir_env:
        raise RuntimeError("TRITON_CACHE_DIR must point to an empty build cache")

    cache_dir = Path(cache_dir_env)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError(f"TRITON_CACHE_DIR must be empty: {cache_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compile_kernels()
    manifest = {
        "format": 2,
        "dtype": "f32",
        "module": MERGED_MODULE_NAME,
        "kernels": [
            copy_artifact(cache_dir, args.output_dir, "flagos_add_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_mul_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_scale_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_copy_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_copy_strided_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_swiglu_split_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_set_rows_f32_f16", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_flash_attn_decode_f32_f16", ATTENTION_BLOCK_N),
            copy_artifact(cache_dir, args.output_dir, "flagos_rope_neox_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_rms_norm_f32", RMS_NORM_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_rms_norm_mul_f32", RMS_NORM_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_cast_f32_f16", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_cast_f16_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_dequant_q4_k_f16", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_dequant_q6_k_f16", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_mul_mat_q4_k_f32", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_mul_mat_q4_k_f32_batched", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_mul_mat_q6_k_f32", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_mul_mat_q6_k_f32_batched", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_get_rows_q4_k_f32", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_get_rows_q6_k_f32", QK_K),
            copy_artifact(cache_dir, args.output_dir, "flagos_get_rows_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_ssm_conv_f32", BLOCK_SIZE),
            copy_artifact(
                cache_dir, args.output_dir, "flagos_ssm_conv_silu_f32",
                SSM_CONV_SILU_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_sub_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_div_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_sigmoid_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_exp_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_softplus_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_fill_f32", BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_sum_rows_f32", ROW_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_l2_norm_f32", ROW_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_norm_f32", ROW_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_cumsum_f32", ROW_BLOCK_SIZE),
            copy_artifact(cache_dir, args.output_dir, "flagos_soft_max_f32", ROW_BLOCK_SIZE),
        ],
    }
    if not args.no_merge:
        build_merged_module(
            cache_dir,
            args.output_dir,
            [entry["name"] for entry in manifest["kernels"]],
            args.sdk_root,
        )
    else:
        manifest.pop("module")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
