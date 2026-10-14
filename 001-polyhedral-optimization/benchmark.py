import torch
import triton
import triton.testing
from golden_kernel import launch_fused_block
from baseline_example import rms_norm_residual_block, HIDDEN_DIM, SEQ_LEN


def run_benchmark(warmup=1, rep=25):
    """
    Benchmark using triton.testing.do_bench for accurate GPU timing.

    Args:
        warmup: Number of warmup iterations (default: 25)
        rep: Number of measurement repetitions (default: 100)
    """
    device, dtype = "cuda", torch.bfloat16

    # Create test tensors
    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    res = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    w = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    # Compile baseline (triggers compilation before benchmarking)
    compiled_baseline = torch.compile(rms_norm_residual_block, fullgraph=True)

    print(f"Benchmarking Inductor (warmup={warmup}, rep={rep})...")
    inductor_ms = triton.testing.do_bench(
        lambda: compiled_baseline(x, res, w),
        warmup=warmup,
        rep=rep
    )

    # Benchmark Golden kernel
    print(f"Benchmarking Golden Kernel (warmup={warmup}, rep={rep})...")
    golden_ms = triton.testing.do_bench(
        lambda: launch_fused_block(x, res, w),
        warmup=warmup,
        rep=rep
    )

    # Print results
    print("\n" + "="*50)
    print(f"H200 BENCHMARK RESULTS (triton.testing.do_bench)")
    print(f"Config: SEQ_LEN={SEQ_LEN}, HIDDEN_DIM={HIDDEN_DIM}")
    print(f"Warmup: {warmup}, Repetitions: {rep}")
    print("="*50)
    print(f"Inductor Baseline (2 kernels): {inductor_ms:.4f} ms")
    print(f"Golden Fused (1 kernel):       {golden_ms:.4f} ms")
    print("-" * 50)
    print(f"Latency Reduction:             {inductor_ms - golden_ms:.4f} ms")
    print(f"Speedup:                       {inductor_ms / golden_ms:.2f}x")
    print("="*50)

    return inductor_ms, golden_ms


if __name__ == "__main__":
    run_benchmark()
