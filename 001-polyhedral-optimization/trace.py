import torch
import triton
import triton.language as tl
from .golden_kernel import launch_fused_block
from .baseline_example import rms_norm_residual_block, HIDDEN_DIM, SEQ_LEN

def benchmark():
    device, dtype = "cuda", torch.bfloat16
    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    res = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    w = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    # Compile Baseline
    compiled_baseline = torch.compile(rms_norm_residual_block, fullgraph=True)
    
    # Warmup
    for _ in range(1):
        _ = compiled_baseline(x, res, w)
        _ = launch_fused_block(x, res, w)

    # Time Inductor
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100): _ = compiled_baseline(x, res, w)
    end_event.record()
    torch.cuda.synchronize()
    inductor_time = start_event.elapsed_time(end_event) / 100

    # Time Golden Kernel
    start_event.record()
    for _ in range(100): _ = launch_fused_block(x, res, w)
    end_event.record()
    torch.cuda.synchronize()
    golden_time = start_event.elapsed_time(end_event) / 100

    print(f"--- H200 Performance Results ---")
    print(f"Inductor (Two Kernels): {inductor_time:.4f} ms")
    print(f"Golden (Polyhedral Fusion): {golden_time:.4f} ms")
    print(f"Speedup: {inductor_time / golden_time:.2f}x")
    
    # Verification
    ref = compiled_baseline(x, res, w)
    tst = launch_fused_block(x, res, w)
    torch.testing.assert_close(ref, tst, atol=1e-2, rtol=1e-2)
    print("Verification: PASSED")

if __name__ == "__main__":
    benchmark()
