import torch
import torch.nn.functional as F
import time

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def benchmark_fp8_mm(m, n, k, dtype=torch.float16, qdtype=torch.float8_e4m3fn, num_iters=1000):
    # create test inputs
    x = torch.randn((m, k), dtype=dtype, device='cuda')
    # Note: cuBLASLt float8 matmul requires column major for the second argument
    w = torch.randn((n, k), dtype=dtype, device='cuda').t()

    x_fp8, x_inv_s = to_float8(x, dtype=qdtype)
    w_fp8, w_inv_s = to_float8(w, dtype=qdtype)

    # Warmup
    for _ in range(10):
        y_fp16 = torch.mm(x, w)
        y, _ = torch._scaled_mm(x_fp8, w_fp8, out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
    torch.cuda.synchronize()
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark float8 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y, _ = torch._scaled_mm(x_fp8, w_fp8, out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
    torch.cuda.synchronize()
    fp8_time = time.perf_counter() - start_time
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark float8 matmul with dynamic activation quant
    start_time = time.perf_counter()
    for _ in range(num_iters):
        x_fp8, x_inv_s = to_float8(x, dtype=qdtype)
        y, _ = torch._scaled_mm(x_fp8, w_fp8, out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
    torch.cuda.synchronize()
    fp8_actquant_time = time.perf_counter() - start_time
    # "Cool down" the GPU in between benchmarks to avoid throttling
    time.sleep(0.2)

    # Benchmark fp16 matmul
    start_time = time.perf_counter()
    for _ in range(num_iters):
        y_fp16 = torch.mm(x, w)
    torch.cuda.synchronize()
    fp16_time = time.perf_counter() - start_time
    

    # Compare output of float8 matmul to the fp16 baseline
    cos_sim = F.cosine_similarity(y_fp16.reshape(-1), y.reshape(-1), dim=0)

    # print(f'Cosine similarity: {cos_sim.item():.4f} | FP8 matmul time:  {fp8_time:.4f} seconds | FP16 matmul time: {fp16_time:.4f} seconds')
    print(f'{m}, {n}, {k}, {cos_sim.item():.4f}, {fp16_time:.4f}, {fp8_time:.4f}, {fp8_actquant_time:.4f}, {fp16_time / fp8_time:.3f}, {fp16_time / fp8_actquant_time:.3f}')


if __name__ == "__main__":
    # layer_sizes = [(512, 512),( 1024, 1024), (2048, 2048), (4096, 4096)]
    layer_sizes = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # batch_sizes = [i+1 for i in range(256)]
    num_iters = 10000

    print("M, N, K, Cosine similarity, FP16 time, FP8 time, FP8 dynamic quant time, Speedup, Speedup dynamic quant")
    for N, K in layer_sizes:
        for M in batch_sizes:
            benchmark_fp8_mm(m=M, n=N, k=K, num_iters=num_iters)