from typing import Optional, Tuple

import torch
import time
import triton
import triton.language as tl

try:
    from vllm._C import ops as vllm_ops
except ImportError:
    pass


def static_scaled_fp8_quant_cuda(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn, device="cuda")
    assert scale is not None
    vllm_ops.static_scaled_fp8_quant(output, input, scale)
    return output, scale


@triton.autotune(
    configs=[
        # triton.Config(kwargs={"BLOCK_SIZE": 32}),
        # triton.Config(kwargs={"BLOCK_SIZE": 64}),
        # triton.Config(kwargs={"BLOCK_SIZE": 128}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
    reset_to_zero=["output_ptr"],
)
@triton.jit
def scaled_fp8_quant_kernel(
    output_ptr, input_ptr, scale_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    scale_val = tl.load(scale_ptr)

    # Scale and convert to FP8 (torch.float8_e4m3fn == tl.float8e4nv)
    output_vals = (input_vals / scale_val).to(tl.float8e4nv)
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 32, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 64, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 128, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "NUM_CHUNKS": 4}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 32, "NUM_CHUNKS": 8}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 64, "NUM_CHUNKS": 8}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 128, "NUM_CHUNKS": 8}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "NUM_CHUNKS": 8}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "NUM_CHUNKS": 8}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "NUM_CHUNKS": 8}, num_stages=3),
    ],
    key=["n_elements"],
    reset_to_zero=["output_ptr"],
)
@triton.jit
def scaled_fp8_quant_kernel_chunked(
    output_ptr,
    input_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    for chunk_id in tl.static_range(NUM_CHUNKS):
        block_start = pid * NUM_CHUNKS * BLOCK_SIZE + chunk_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        input_vals = tl.load(input_ptr + offsets, mask=mask)
        scale_val = tl.load(scale_ptr)

        # Scale and convert to FP8 (torch.float8_e4m3fn == tl.float8e4nv)
        output_vals = (input_vals / scale_val).to(tl.float8e4nv)
        tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128, "stride": 128}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "stride": 128}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "stride": 128}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "stride": 128}, num_stages=2),
        triton.Config(kwargs={"BLOCK_SIZE": 128, "stride": 128}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "stride": 128}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "stride": 128}, num_stages=3),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "stride": 128}, num_stages=3),
    ],
    key=["n_elements"],
    reset_to_zero=["output_ptr"],
)
@triton.jit
def scaled_fp8_quant_kernel_unroll(
    output_ptr,
    input_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    stride: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    for offset in tl.range(0, BLOCK_SIZE, stride, num_stages=8):
        offsets = block_start + offset + tl.arange(0, stride)
        mask = offsets < n_elements

        input_vals = tl.load(input_ptr + offsets, mask=mask)
        scale_val = tl.load(scale_ptr)

        # Scale and convert to FP8 (torch.float8_e4m3fn == tl.float8e4nv)
        output_vals = (input_vals / scale_val).to(tl.float8e4nv)
        tl.store(output_ptr + offsets, output_vals, mask=mask)


def static_scaled_fp8_quant_triton(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn, device="cuda")

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scaled_fp8_quant_kernel[grid](output, input, scale, n_elements)

    return output, scale


iters = 100
original_dtype = torch.float16
shapes = [
    (2**10, 2**10),
    (2**11, 2**11),
    (2**12, 2**12),
    (2**13, 2**13),
    (2**14, 2**14),
    (2**15, 2**15),
]

for shape in shapes:
    input = torch.rand(shape, dtype=original_dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Warmup
    triton_output = static_scaled_fp8_quant_triton(input, scale)
    cuda_output = static_scaled_fp8_quant_cuda(input, scale)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(iters):
        triton_output, _ = static_scaled_fp8_quant_triton(input, scale)
    torch.cuda.synchronize()
    triton_dur = (time.perf_counter() - start_time) / iters

    start_time = time.perf_counter()
    for _ in range(iters):
        cuda_output, _ = static_scaled_fp8_quant_cuda(input, scale)
    torch.cuda.synchronize()
    cuda_dur = (time.perf_counter() - start_time) / iters

    # print("cuda_output:", cuda_output)
    # print("triton_output:", triton_output)

    # print(f"CUDA quant took: {cuda_dur}")
    # print(f"Triton quant took: {triton_dur}")

    print(
        f"{shape[0]}, {shape[1]}, {cuda_dur/triton_dur:.2f}x, {triton_dur*1000000:.4f}us"
    )

    assert torch.allclose(
        triton_output.to(torch.float16), cuda_output.to(torch.float16)
    )
