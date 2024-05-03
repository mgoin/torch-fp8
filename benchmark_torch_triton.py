import torch
import torch.nn.functional as F
import time
from triton_gemm_split_k import gemm_split_k
import triton.ops
import triton.language as tl


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def benchmark_fp8_mm(m, n, k, dtype=torch.float16, qdtype=torch.float8_e4m3fn, num_iters=1000):
    # create test inputs
    x = torch.randn((m, k), dtype=dtype, device='cuda')
    # Note: cuBLASLt float8 matmul requires column major for the second argument
    w = torch.randn((n, k), dtype=dtype, device='cuda').T

    # Used for scaled_mm
    x_fp8_scaled, x_inv_s = to_float8(x, dtype=qdtype)
    w_fp8_scaled, w_inv_s = to_float8(w, dtype=qdtype)

    # Used for triton kernel
    x_fp8 = x.to(qdtype)
    w_fp8 = w.to(qdtype)
    x_fp8 = triton.reinterpret(x_fp8, tl.float8e4nv)
    w_fp8 = triton.reinterpret(w_fp8, tl.float8e4nv)

    # Warmup
    for _ in range(10):
        y_fp16 = torch.mm(x, w)
        y_torch, _ = torch._scaled_mm(x_fp8_scaled, w_fp8_scaled, out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
        y_triton = triton.ops.matmul(x_fp8, w_fp8)
        y_triton_split = gemm_split_k(x_fp8_scaled, w_fp8_scaled, scale_a=x_inv_s.item(), scale_b=w_inv_s.item())
    torch.cuda.synchronize()
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark float8 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y_torch, _ = torch._scaled_mm(x_fp8_scaled, w_fp8_scaled, out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
    torch.cuda.synchronize()
    fp8_time = time.perf_counter() - start_time
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark triton float8 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y_triton = triton.ops.matmul(x_fp8, w_fp8)
    torch.cuda.synchronize()
    fp8_triton_time = time.perf_counter() - start_time
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark triton split k float8 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y_triton_split = gemm_split_k(x_fp8_scaled, w_fp8_scaled, scale_a=x_inv_s.item(), scale_b=w_inv_s.item())
    torch.cuda.synchronize()
    fp8_triton_split_time = time.perf_counter() - start_time
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark fp16 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y_fp16 = torch.mm(x, w)
    torch.cuda.synchronize()
    fp16_time = time.perf_counter() - start_time
    

    # Compare output of float8 matmul to the fp16 baseline
    cos_sim_torch = F.cosine_similarity(y_fp16.reshape(-1), y_torch.reshape(-1), dim=0)
    cos_sim_triton = F.cosine_similarity(y_fp16.reshape(-1), y_triton.reshape(-1), dim=0)
    cos_sim_triton_split = F.cosine_similarity(y_fp16.reshape(-1), y_triton_split.reshape(-1), dim=0)

    # print(f'Cosine similarity: {cos_sim.item():.4f} | FP8 matmul time:  {fp8_time:.4f} seconds | FP16 matmul time: {fp16_time:.4f} seconds')
    print(f'{m}, {n}, {k}, {cos_sim_torch.item():.4f}, {cos_sim_triton.item():.4f}, {cos_sim_triton_split.item():.4f}, {fp16_time:.4f}, {fp8_time:.4f}, {fp8_triton_time:.4f}, {fp8_triton_split_time:.4f}, {fp16_time / fp8_time:.3f}, {fp16_time / fp8_triton_time:.3f}, {fp16_time / fp8_triton_split_time:.3f}')


if __name__ == "__main__":
    # layer_sizes = [(512, 512),( 1024, 1024), (2048, 2048), (4096, 4096)]
    layer_sizes = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
    ]
    layer_sizes = [(8192, 8192)]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # batch_sizes = [i+1 for i in range(256)]
    num_iters = 10000

    print("M, N, K, Cosine similarity torch, Cosine similarity triton, Cosine similarity triton split, FP16 time, FP8 torch time, FP8 triton time, FP8 triton split time, Speedup torch, Speedup triton, Speedup triton split")
    for N, K in layer_sizes:
        for M in batch_sizes:
            benchmark_fp8_mm(m=M, n=N, k=K, num_iters=num_iters)