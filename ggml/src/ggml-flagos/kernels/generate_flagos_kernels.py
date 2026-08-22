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
# Decode GEMV assigns one program to one output row. On KS20-A, using one warp
# lets each thread process eight lanes and is materially faster than scheduling
# four or eight warps for the same 256-value reduction. Keep this separate from
# attention, reductions, and batched GEMM, which retain NUM_WARPS.
GEMV_NUM_WARPS = 1
# Must be >= the widest row we accept in supports_op. Qwen3.5-9B has a
# 4096-wide hidden state; at 2048 every RMS norm was declined to the CPU.
RMS_NORM_BLOCK_SIZE = 4096
# Columns processed per program by the batched (prefill) quantized GEMMs.
# The dequantized weight row is reused across the whole tile.
MUL_MAT_COLS_PER_BLOCK = 16
# Row-wise kernels use one program per row with BLOCK spanning the whole row.
ROW_BLOCK_SIZE = 1024
QK_K = 256
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
def flagos_mul_f32(x, y, output, n_elements, y_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y_offsets = offsets % y_elements
    values = tl.load(x + offsets, mask=mask) * tl.load(y + y_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


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


def compile_kernels() -> None:
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
    torch.testing.assert_close(output, x + y_add)

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
    torch.testing.assert_close(output, expected)

    flagos_scale_f32[grid](
        x,
        output,
        0.75,
        -0.25,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, x * 0.75 - 0.25)

    flagos_copy_f32[grid](
        x,
        output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, x)

    broadcast_index = torch.arange(n_elements, device="cuda") % y_elements

    flagos_sub_f32[grid](
        x, y_mul, output, n_elements, y_elements,
        BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, x - y_mul[broadcast_index])

    # Keep the divisor away from zero so the reference stays finite.
    y_div = y_mul.abs() + 0.5
    flagos_div_f32[grid](
        x, y_div, output, n_elements, y_elements,
        BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, x / y_div[broadcast_index])

    flagos_sigmoid_f32[grid](
        x, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, torch.sigmoid(x))

    flagos_exp_f32[grid](
        x, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, torch.exp(x))

    # Include a large value to exercise the overflow-avoiding branch.
    softplus_input = torch.cat([x[:-1], torch.tensor([40.0], device="cuda")])
    flagos_softplus_f32[grid](
        softplus_input, output, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, torch.nn.functional.softplus(softplus_input))

    flagos_fill_f32[grid](
        output, 0.375, n_elements, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, torch.full_like(output, 0.375))

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
    torch.testing.assert_close(sums, rows.sum(dim=1))

    flagos_l2_norm_f32[row_grid](
        rows, row_output, row_cols, 1e-12, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(
        row_output,
        rows / rows.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12))

    flagos_norm_f32[row_grid](
        rows, row_output, row_cols, 1e-5, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    centered = rows - rows.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        row_output, centered / (centered.pow(2).mean(dim=1, keepdim=True) + 1e-5).sqrt())

    flagos_cumsum_f32[row_grid](
        rows, row_output, row_cols, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    # The tree scan and PyTorch reference accumulate in different orders.
    torch.testing.assert_close(
        row_output, rows.cumsum(dim=1), rtol=2e-5, atol=2e-5)

    soft_max_mask = torch.randn(row_count, row_cols, device="cuda", dtype=torch.float32)
    flagos_soft_max_f32[row_grid](
        rows, soft_max_mask, row_output, row_cols, 0.125,
        HAS_MASK=True, BLOCK=row_cols, num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(
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
    torch.testing.assert_close(strided_dst, expected_strided)

    flagos_swiglu_split_f32[grid](
        x,
        y_add,
        output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(output, torch.nn.functional.silu(x) * y_add)

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
    torch.testing.assert_close(set_rows_output[set_rows_index].float(), set_rows_input, rtol=5e-4, atol=5e-4)

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
    torch.testing.assert_close(
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
        torch.testing.assert_close(attention_output, attention_expected, rtol=8e-3, atol=8e-3)

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
    torch.testing.assert_close(rope_output, rope_expected.reshape(-1), rtol=1e-4, atol=5e-4)

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
    torch.testing.assert_close(norm_output, expected_norm, rtol=2e-5, atol=2e-5)

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
    torch.testing.assert_close(norm_output, expected_norm, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(mul_output, expected_norm * weight, rtol=2e-5, atol=2e-5)

    # GET_ROWS over a Q4_K table.
    rows_table = 12
    rows_blocks = 3
    rows_cols = rows_blocks * QK_K
    table_packed, table_dequantized = make_q4_k_weights(rows_table, rows_blocks)
    row_ids = torch.tensor([7, 0, 11, 3], device="cuda", dtype=torch.int32)
    get_rows_output = torch.empty(
        (row_ids.numel(), rows_cols), device="cuda", dtype=torch.float32
    )
    flagos_get_rows_q4_k_f32[(row_ids.numel(), rows_blocks)](
        table_packed,
        table_packed.view(torch.float16),
        row_ids,
        get_rows_output,
        rows_cols,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(
        get_rows_output, table_dequantized[row_ids.long()], rtol=2e-3, atol=2e-3
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
    torch.testing.assert_close(
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
    torch.testing.assert_close(dense_output, dense_table[row_ids.long()])

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
    torch.testing.assert_close(conv_output, conv_expected, rtol=2e-5, atol=2e-5)

    cast_output = torch.empty(n_elements, device="cuda", dtype=torch.float16)
    flagos_cast_f32_f16[grid](
        x,
        cast_output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(cast_output.float(), x.half().float())

    cast_back_output = torch.empty(n_elements, device="cuda", dtype=torch.float32)
    flagos_cast_f16_f32[grid](
        cast_output,
        cast_back_output,
        n_elements,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(cast_back_output, x.half().float())

    mat_rows = 17
    mat_blocks = 6
    mat_k = mat_blocks * QK_K
    activation = torch.randn(mat_k, device="cuda", dtype=torch.float32)
    mat_output = torch.empty(mat_rows, device="cuda", dtype=torch.float32)

    q4_packed, q4_dequantized = make_q4_k_weights(mat_rows, mat_blocks)
    q4_dequantized_f16 = torch.empty_like(q4_dequantized, dtype=torch.float16)
    flagos_dequant_q4_k_f16[(mat_rows * mat_blocks,)](
        q4_packed,
        q4_packed.view(torch.float16),
        q4_dequantized_f16,
        num_warps=NUM_WARPS,
    )
    torch.testing.assert_close(q4_dequantized_f16.float(), q4_dequantized.half().float())
    flagos_mul_mat_q4_k_f32[(mat_rows,)](
        q4_packed,
        q4_packed.view(torch.float16),
        activation,
        mat_output,
        mat_k,
        mat_rows,
        num_warps=GEMV_NUM_WARPS,
    )
    torch.testing.assert_close(mat_output, q4_dequantized @ activation, rtol=2e-5, atol=2e-4)

    # Test batched (multi-column) GEMM for prefill
    # Deliberately not a multiple of MUL_MAT_COLS_PER_BLOCK so the tail path
    # in the batched GEMMs is exercised; 32 columns hid a tail bug entirely.
    mat_columns = 37
    # ggml layout: each of the `mat_columns` activation columns is contiguous
    # along k, so store them as rows here and transpose in the reference.
    activation_batched = torch.randn(mat_columns, mat_k, device="cuda", dtype=torch.float32)
    mat_output_batched = torch.zeros(mat_columns * mat_rows, device="cuda", dtype=torch.float32)
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
    torch.testing.assert_close(
        mat_output_batched.view(mat_columns, mat_rows).T,
        q4_dequantized @ activation_batched.T,
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
    torch.testing.assert_close(q6_dequantized_f16.float(), q6_dequantized.half().float())

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
    torch.testing.assert_close(
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
    torch.testing.assert_close(mat_output, q6_dequantized @ activation, rtol=2e-5, atol=2e-3)
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
