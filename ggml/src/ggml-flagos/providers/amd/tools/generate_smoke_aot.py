#!/usr/bin/env python3
"""Generate a tiny AMD HSACO package for the FlagOS provider smoke tests.

This is deliberately a test artifact generator, not the Qwen kernel pipeline.
It uses the installed FlagTree/Triton 3.6 compiler and records the same
metadata consumed by the C++ HIP module registry.
"""

import argparse
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.language.extra import libdevice


@triton.jit
def flagos_add_f32(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def double(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def flagos_get_rows_q4_k_f32(weights_u8, weights_f16, row_index, output, n_cols):
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
    scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
    scale = tl.where(group < 4, scale_lo & 63,
                    (scale_lo & 15) | ((scale_hi >> 6) << 4)).to(tl.float32)
    min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
    min_hi = tl.load(weights_u8 + block_byte + 4 + group)
    minimum = tl.where(group < 4, min_lo & 63,
                       (min_lo >> 4) | ((min_hi >> 6) << 4)).to(tl.float32)
    within_64 = lanes % 64
    quant_byte = tl.load(weights_u8 + block_byte + 16 + (lanes // 64) * 32 + within_64 % 32)
    quant = tl.where(within_64 < 32, quant_byte & 15, quant_byte >> 4).to(tl.float32)
    tl.store(output + token * n_cols + sub * 256 + lanes,
             d * scale * quant - dmin * minimum)


@triton.jit
def flagos_get_rows_q6_k_f32(weights_u8, weights_f16, row_index, output, n_cols):
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
def flagos_rms_norm_f32(output, x, n_cols, eps, BLOCK: tl.constexpr):
    # Keep this signature and launch geometry identical to the FlagTree
    # kernel used by the full FlagOS package.  One Triton program owns one
    # contiguous row and BLOCK is the compiled row width.
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
def flagos_add_rms_norm_mul_f32(
    norm_output,
    mul_output,
    x,
    bias,
    weight,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    values += tl.load(bias + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(norm_output + row * n_cols + cols, normalized, mask=mask)
    tl.store(mul_output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_swiglu_split_f32(gate, up, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate_values = tl.load(gate + offsets, mask=mask, other=0.0)
    up_values = tl.load(up + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, gate_values * tl.sigmoid(gate_values) * up_values, mask=mask)


@triton.jit(do_not_specialize=["q_per_kv"])
def flagos_flash_attn_decode_f32_f16(
    q, k, v, attention_mask, output, key_length, q_per_kv,
    stride_q_token, stride_q_head, stride_k_token, stride_k_head,
    stride_v_token, stride_v_head, stride_output_token, stride_output_head,
    scale, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    query_head = tl.program_id(0)
    kv_head = query_head // q_per_kv
    query_rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    valid_query = query_rows == 0
    q_block = tl.load(q + query_rows[:, None] * stride_q_token + query_head * stride_q_head + dims[None, :],
                      mask=valid_query[:, None], other=0.0).to(tl.float16)
    running_max = tl.where(valid_query, -float("inf"), 0.0)
    running_sum = tl.where(valid_query, 0.0, 1.0)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for key_start in tl.range(0, key_length, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        k_block = tl.load(k + dims[:, None] + kv_head * stride_k_head + key_offsets[None, :] * stride_k_token,
                          mask=key_mask[None, :], other=0.0)
        scores = tl.dot(q_block, k_block) * scale
        scores += tl.load(attention_mask + key_offsets[None, :], mask=key_mask[None, :], other=-float("inf"))
        valid = valid_query[:, None] & key_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        block_max = tl.where(valid_query, block_max, 0.0)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probabilities = tl.where(valid, tl.exp(scores - new_max[:, None]), 0.0)
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= rescale[:, None]
        v_block = tl.load(v + key_offsets[:, None] * stride_v_token + kv_head * stride_v_head + dims[None, :],
                          mask=key_mask[:, None], other=0.0)
        accumulator += tl.dot(probabilities.to(tl.float16), v_block)
        running_sum = running_sum * rescale + block_sum
        running_max = new_max
    normalized = accumulator / running_sum[:, None]
    tl.store(output + query_rows[:, None] * stride_output_token + query_head * stride_output_head + dims[None, :],
             normalized, mask=valid_query[:, None])


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
def flagos_rope_kv_store_f32_f16(
    x,
    positions,
    row_index,
    output,
    n_cols,
    n_rows,
    n_dst_rows,
    head_dim,
    n_heads,
    n_dims,
    freq_base,
    freq_scale,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_cols * n_rows
    rows = offsets // n_cols
    cols = offsets % n_cols
    destination_rows = tl.load(row_index + rows, mask=mask, other=0)
    mask = mask & (destination_rows >= 0) & (destination_rows < n_dst_rows)
    head = cols // head_dim
    dim = cols % head_dim
    source = rows * n_cols + head * head_dim
    half = n_dims // 2
    rotated = dim < n_dims
    pair_dim = tl.where(dim < half, dim + half, dim - half)
    x0 = tl.load(x + source + tl.where(dim < half, dim, pair_dim), mask=mask, other=0.0)
    x1 = tl.load(x + source + tl.where(dim < half, pair_dim, dim), mask=mask, other=0.0)
    position = tl.load(positions + rows, mask=mask, other=0).to(tl.float32)
    exponent = -2.0 * (dim % half).to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    cos_theta = libdevice.cos(theta)
    sin_theta = libdevice.sin(theta)
    rotated_value = tl.where(dim < half, x0 * cos_theta - x1 * sin_theta,
                             x1 * cos_theta + x0 * sin_theta)
    values = tl.where(rotated, rotated_value, tl.load(x + source + dim, mask=mask, other=0.0))
    tl.store(output + destination_rows * n_cols + cols, values, mask=mask)


@triton.jit
def flagos_silu_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, values * tl.sigmoid(values), mask=mask)


@triton.jit
def flagos_set_rows_f32_f16(x, row_index, output, n_cols, n_rows, n_dst_rows, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_cols * n_rows
    rows = offsets // n_cols
    cols = offsets % n_cols
    destination_rows = tl.load(row_index + rows, mask=mask, other=0)
    mask = mask & (destination_rows >= 0) & (destination_rows < n_dst_rows)
    values = tl.load(x + rows * n_cols + cols, mask=mask, other=0.0)
    tl.store(output + destination_rows * n_cols + cols, values, mask=mask)


@triton.jit
def flagos_mul_mat_q4_k_f32(weights_u8, weights_f16, x, output, k, rows):
    row = tl.program_id(0)
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
        low_scale_byte = tl.load(weights_u8 + block_byte + 4 + tl.where(low_group < 4, low_group, low_group + 4))
        low_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(low_group - 4, 0))
        low_scale = tl.where(low_group < 4, low_scale_byte & 63,
                             (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
        low_minimum = tl.where(low_group < 4, low_min_byte & 63,
                               (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
        high_scale_byte = tl.load(weights_u8 + block_byte + 4 + tl.where(high_group < 4, high_group, high_group + 4))
        high_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(high_group - 4, 0))
        high_scale = tl.where(high_group < 4, high_scale_byte & 63,
                              (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
        high_minimum = tl.where(high_group < 4, high_min_byte & 63,
                                (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
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
        accumulator += d * (s0 * q0.to(tl.float32) * a0 + s1 * q1.to(tl.float32) * a1
                            + s2 * q2.to(tl.float32) * a2 + s3 * q3.to(tl.float32) * a3)
    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def flagos_mul_mat_q4_k_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK), columns - 1)
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
        scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(group - 4, 0))
        scale = tl.where(group < 4, scale_lo & 63,
                         (scale_lo & 15) | ((scale_hi >> 6) << 4)).to(tl.float32)
        min_lo = tl.load(weights_u8 + block_byte + 4 + group + 4)
        min_hi = tl.load(weights_u8 + block_byte + 4 + group)
        minimum = tl.where(group < 4, min_lo & 63,
                           (min_lo >> 4) | ((min_hi >> 6) << 4)).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 16 + (lanes // 64) * 32 + lanes % 32)
        quant = tl.where(lanes % 64 < 32, quant_byte & 15, quant_byte >> 4).to(tl.float32)
        values = d * scale * quant - dmin * minimum
        activations = tl.load(x + cols[:, None] * k + block * 256 + lanes[None, :]).to(tl.float32)
        accumulator += values[None, :] * activations
    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


@triton.jit
def flagos_mul_mat_q6_k_f32_batched(weights_u8, weights_f16, x, output, k, rows, columns,
                                    COLS_PER_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = tl.minimum(col_block * COLS_PER_BLOCK + tl.arange(0, COLS_PER_BLOCK), columns - 1)
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
        activations = tl.load(x + cols[:, None] * k + block * 256 + lanes[None, :]).to(tl.float32)
        accumulator += values[None, :] * activations
    tl.store(output + cols * rows + row, tl.sum(accumulator, axis=1))


def compile_kernel(fn, signature, block_size, target, constants=None):
    constexprs = {"BLOCK": block_size} if "BLOCK" in signature else {}
    constexprs.update(constants or {})
    source = ASTSource(fn, signature, constexprs)
    return triton.compile(source, target=target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    # Match the full FlagOS RMSNorm package width so the generated smoke
    # artifact can exercise model-sized hidden states (including Qwen3-1.7B).
    parser.add_argument("--block-size", type=int, default=4096)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        raise RuntimeError(f"expected HIP target, got {target}")

    entries = []
    kernels = [
        (flagos_add_f32, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}, "flagos_add_f32"),
        (double, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}, "double"),
        (flagos_rms_norm_f32, {"output": "*fp32", "x": "*fp32", "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr"}, "flagos_rms_norm_f32"),
        (flagos_rms_norm_mul_f32, {
            "norm_output": "*fp32", "mul_output": "*fp32", "x": "*fp32", "weight": "*fp32",
            "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr",
        }, "flagos_rms_norm_mul_f32"),
        (flagos_add_rms_norm_mul_f32, {
            "norm_output": "*fp32", "mul_output": "*fp32", "x": "*fp32", "bias": "*fp32", "weight": "*fp32",
            "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr",
        }, "flagos_add_rms_norm_mul_f32"),
        (flagos_swiglu_split_f32, {
            "gate": "*fp32", "up": "*fp32", "output": "*fp32", "n_elements": "i32", "BLOCK": "constexpr",
        }, "flagos_swiglu_split_f32"),
        (flagos_flash_attn_decode_f32_f16, {
            "q": "*fp32", "k": "*fp16", "v": "*fp16", "attention_mask": "*fp16", "output": "*fp32",
            "key_length": "i32", "q_per_kv": "i32", "stride_q_token": "i32", "stride_q_head": "i32",
            "stride_k_token": "i32", "stride_k_head": "i32", "stride_v_token": "i32", "stride_v_head": "i32",
            "stride_output_token": "i32", "stride_output_head": "i32", "scale": "fp32",
            "HEAD_DIM": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
        }, "flagos_flash_attn_decode_f32_f16", {"HEAD_DIM": 128, "BLOCK_M": 16, "BLOCK_N": 32}),
        (flagos_rope_neox_f32, {
            "x": "*fp32", "positions": "*i32", "output": "*fp32",
            "n_elements": "i32", "ne0": "i32", "ne1": "i32", "ne2": "i32",
            "n_dims": "i32", "freq_base": "fp32", "freq_scale": "fp32",
            "BLOCK": "constexpr",
        }, "flagos_rope_neox_f32"),
        (flagos_rope_kv_store_f32_f16, {
            "x": "*fp32", "positions": "*i32", "row_index": "*i64", "output": "*fp16",
            "n_cols": "i32", "n_rows": "i32", "n_dst_rows": "i32", "head_dim": "i32",
            "n_heads": "i32", "n_dims": "i32", "freq_base": "fp32", "freq_scale": "fp32",
            "BLOCK": "constexpr",
        }, "flagos_rope_kv_store_f32_f16"),
        (flagos_silu_f32, {"x": "*fp32", "output": "*fp32", "n_elements": "i32", "BLOCK": "constexpr"}, "flagos_silu_f32"),
        (flagos_set_rows_f32_f16, {
            "x": "*fp32", "row_index": "*i64", "output": "*fp16",
            "n_cols": "i32", "n_rows": "i32", "n_dst_rows": "i32", "BLOCK": "constexpr",
        }, "flagos_set_rows_f32_f16"),
        (flagos_get_rows_q4_k_f32, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "row_index": "*i32",
            "output": "*fp32", "n_cols": "i32",
        }, "flagos_get_rows_q4_k_f32"),
        (flagos_get_rows_q6_k_f32, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "row_index": "*i32",
            "output": "*fp32", "n_cols": "i32",
        }, "flagos_get_rows_q6_k_f32"),
        (flagos_mul_mat_q4_k_f32, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32", "output": "*fp32",
            "k": "i32", "rows": "i32",
        }, "flagos_mul_mat_q4_k_f32"),
        (flagos_mul_mat_q6_k_f32, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32", "output": "*fp32",
            "k": "i32", "rows": "i32",
        }, "flagos_mul_mat_q6_k_f32"),
        (flagos_mul_mat_q4_k_f32_batched, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32", "output": "*fp32",
            "k": "i32", "rows": "i32", "columns": "i32", "COLS_PER_BLOCK": "constexpr",
        }, "flagos_mul_mat_q4_k_f32_batched", {"COLS_PER_BLOCK": 16}),
        (flagos_mul_mat_q6_k_f32_batched, {
            "weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32", "output": "*fp32",
            "k": "i32", "rows": "i32", "columns": "i32", "COLS_PER_BLOCK": "constexpr",
        }, "flagos_mul_mat_q6_k_f32_batched", {"COLS_PER_BLOCK": 16}),
    ]
    for item in kernels:
        fn, signature, name = item[:3]
        constants = item[3] if len(item) > 3 else None
        compiled = compile_kernel(fn, signature, args.block_size, target, constants)
        output = args.output_dir / f"{name}.hsaco"
        output.write_bytes(compiled.asm["hsaco"])
        metadata = compiled.metadata
        entries.append({
            "name": name,
            "symbol": name,
            "file": output.name,
            "shared": metadata.shared,
            "num_warps": metadata.num_warps,
            "warp_size": metadata.warp_size,
            "block_size": (constants or {}).get("BLOCK_N", (constants or {}).get("COLS_PER_BLOCK", args.block_size)),
            "profile_scratch_size": metadata.profile_scratch_size,
            "profile_scratch_align": metadata.profile_scratch_align,
        })

    (args.output_dir / "manifest.json").write_text(json.dumps({
        "format": 2,
        "arch": target.arch,
        "kernels": entries,
    }, indent=2) + "\n")
    print(f"wrote {len(entries)} kernels for {target.backend}:{target.arch} to {args.output_dir}")


if __name__ == "__main__":
    main()
