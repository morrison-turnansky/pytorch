import torch
import triton
import triton.language as tl
import time
from .golden_kernel import launch_fused_block
from .baseline_example import rms_norm_residual_block, HIDDEN_DIM, SEQ_LEN

# --- WALL CLOCK BENCHMARK ---
def run_benchmark(iters=10):
    device, dtype = "cuda", torch.bfloat16
    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    res = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    w = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    # Prepare Baseline (Trigger compilation before timing)
    compiled_baseline = torch.compile(rms_norm_residual_block, fullgraph=True)

    for _ in range(1):
        _ = compiled_baseline(x, res, w)
        _ = launch_fused_block(x, res, w)
    torch.cuda.synchronize()

    # Benchmark Torch.Compile (Inductor)
    print(f"Benchmarking Inductor (Wall Clock)...")
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = compiled_baseline(x, res, w)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    inductor_total = (t1 - t0) * 1000 # ms
    
    # Benchmark Golden Kernel
    print(f"Benchmarking Golden Kernel (Wall Clock)...")
    t2 = time.perf_counter()
    for _ in range(iters):
        _ = launch_fused_block(x, res, w)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    golden_total = (t3 - t2) * 1000 # ms

    # Results
    avg_inductor = inductor_total / iters
    avg_golden = golden_total / iters
    
    print("\n" + "="*30)
    print(f"H200 WALL CLOCK RESULTS")
    print(f"Iterations: {iters}")
    print("="*30)
    print(f"Inductor Baseline: {avg_inductor:.4f} ms/iter")
    print(f"Golden Fused:      {avg_golden:.4f} ms/iter")
    print("-" * 30)
    print(f"LATENCY REDUCTION: {avg_inductor - avg_golden:.4f} ms")
    print(f"SPEEDUP:           {avg_inductor / avg_golden:.2f}x")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
