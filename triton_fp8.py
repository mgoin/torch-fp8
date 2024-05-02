# Taken from https://github.com/pytorch-labs/applied-ai/blob/5297df2b233c467ff7e8748003a730462289e856/kernels/triton/inference/fp8/splitk_gemm_fp8.py

import torch
import triton
import triton.language as tl
import time
import os
os.environ['ENABLE_TMA'] = '1'

@triton.jit
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit()
def col_major(pid,
              m, n,
              block_m: tl.constexpr, block_n: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(a_ptr, b_ptr, c_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            m, n, k,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            split_k: tl.constexpr, group_m: tl.constexpr):

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k*split_k)

    pid_m, pid_n = grouped_launch(pid,
                                  m, n,
                                  block_m, block_n, group_m)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):

        k_remaining = k - k_ * (block_k * split_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk

    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(c_ptrs, acc, mask=mask)

def gemm_split_k(a, b):

    m, k = a.shape
    _, n = b.shape

    block_m = 64
    block_n = 64
    block_k = 512
    num_stages = 3
    num_warps = 8
    split_k = 4
    group_m = 8

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    k = gemm_split_k_kernel[grid](a, b, c,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              m, n, k,
                              block_m, block_n, block_k,
                              split_k, group_m, num_stages=num_stages, num_warps=num_warps)

    return c


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

dtype = torch.float16
qdtype = torch.float8_e4m3fn
m = 16
n = 8192
k = 8192

print(f"Running with M={m}, N={n}, K={k}")

# create test inputs
x = torch.randn((m, k), dtype=dtype, device='cuda')
w = torch.randn((k, n), dtype=dtype, device='cuda')

x_fp8_scaled, x_inv_s = to_float8(x, dtype=qdtype)
w_fp8_scaled, w_inv_s = to_float8(w, dtype=qdtype)

x_fp8 = x.to(qdtype)
w_fp8 = w.T.to(qdtype)

y_torch, _ = torch._scaled_mm(x_fp8_scaled, w_fp8_scaled.t(), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
y_triton = gemm_split_k(x_fp8, w_fp8)
y_fp16 = torch.nn.functional.linear(x, w)

print("y_torch:", y_torch)
print("y_triton:", y_triton)
print("y_fp16:", y_fp16)

print("fp16 vs torch cos_sim:", torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_torch.reshape(-1), dim=0))
print("fp16 vs triton cos_sim:", torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_triton.reshape(-1), dim=0))
