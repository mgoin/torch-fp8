from typing import Optional, Tuple

import torch
import time
import triton
import triton.language as tl

try:
    from vllm._C import cache_ops as vllm_cache_ops
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

@triton.autotune(configs=[
    triton.Config(kwargs={"BLOCK_SIZE": 32}, num_stages=4),
    triton.Config(kwargs={"BLOCK_SIZE": 64}, num_stages=4),
    triton.Config(kwargs={"BLOCK_SIZE": 128}, num_stages=4),
    triton.Config(kwargs={"BLOCK_SIZE": 512}, num_stages=4),
    triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=4),
    triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_stages=4),
  ],
  key=["n_elements"],
  reset_to_zero=["output_ptr"],
)
@triton.jit
def scaled_fp8_quant_kernel(output_ptr, input_ptr, scale_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    scale_val = tl.load(scale_ptr)

    # Scale and convert to FP8 (torch.float8_e4m3fn == tl.float8e4nv)
    output_vals = (input_vals / scale_val).to(tl.float8e4nv)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@triton.autotune(configs=[
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
def scaled_fp8_quant_kernel_chunked(output_ptr, input_ptr, scale_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr, NUM_CHUNKS: tl.constexpr):
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


@triton.autotune(configs=[
    triton.Config(kwargs={"BLOCK_SIZE": 32, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 64, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 128, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 512, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 1024, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 2048, "UNROLL_FACTOR": 2}, num_stages=2),
    triton.Config(kwargs={"BLOCK_SIZE": 32, "UNROLL_FACTOR": 8}, num_stages=3),
    triton.Config(kwargs={"BLOCK_SIZE": 64, "UNROLL_FACTOR": 8}, num_stages=3),
    triton.Config(kwargs={"BLOCK_SIZE": 128, "UNROLL_FACTOR": 8}, num_stages=3),
    triton.Config(kwargs={"BLOCK_SIZE": 512, "UNROLL_FACTOR": 8}, num_stages=3),
    triton.Config(kwargs={"BLOCK_SIZE": 1024, "UNROLL_FACTOR": 8}, num_stages=3),
    triton.Config(kwargs={"BLOCK_SIZE": 2048, "UNROLL_FACTOR": 8}, num_stages=3),
  ],
  key=["n_elements"],
  reset_to_zero=["output_ptr"],
)
@triton.jit
def scaled_fp8_quant_kernel_unroll(output_ptr, input_ptr, scale_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr, UNROLL_FACTOR: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    for _ in tl.static_range(UNROLL_FACTOR):
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
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]*meta["NUM_CHUNKS"]),)
    scaled_fp8_quant_kernel_chunked[grid](output, input, scale, n_elements)

    return output, scale


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
    for _ in range(100):
        triton_output, _ = static_scaled_fp8_quant_triton(input, scale)
    torch.cuda.synchronize()
    triton_dur = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for _ in range(100):
        cuda_output, _ = static_scaled_fp8_quant_cuda(input, scale)
    torch.cuda.synchronize()
    cuda_dur = time.perf_counter() - start_time


    # print("cuda_output:", cuda_output)
    # print("triton_output:", triton_output)

    # print(f"CUDA quant took: {cuda_dur}")
    # print(f"Triton quant took: {triton_dur}")
    print(f"{shape[0]}, {shape[1]}, {cuda_dur/triton_dur:.2f}x")

    assert torch.allclose(triton_output.to(torch.float16), cuda_output.to(torch.float16))