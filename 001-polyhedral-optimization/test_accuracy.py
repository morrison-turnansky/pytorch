"""
Test accuracy of torch implementation from baseline_example and golden kernel
"""
import torch
import pytest
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

from baseline_example import rms_norm_residual_block, HIDDEN_DIM, SEQ_LEN
from golden_kernel import launch_fused_block


def test_torch_kernel():
    """
    Test that compiled torch kernel baseline_example matches up to reasonable
    numerical error with eager call. Use torch.allclose.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA required for this test")

    dtype = torch.bfloat16

    # Create test inputs
    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    residual = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    weight = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    # Run eager (uncompiled) version
    eager_output = rms_norm_residual_block(x, residual, weight)

    # Run compiled version
    compiled_fn = torch.compile(rms_norm_residual_block, fullgraph=True)
    compiled_output = compiled_fn(x, residual, weight)

    # Verify outputs match
    max_diff = (eager_output - compiled_output).abs().max().item()
    mean_diff = (eager_output - compiled_output).abs().mean().item()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # bfloat16 has limited precision, so we use relaxed tolerances
    torch.testing.assert_close(
        eager_output,
        compiled_output,
        atol=1e-1,  # Relaxed for bfloat16
        rtol=5e-2,
        msg=f"Compiled torch kernel does not match eager execution (max_diff={max_diff:.6f})"
    )

    print(f"✓ Torch kernel test passed")


def test_golden_kernel():
    """
    Verify that golden kernel runs and matches output up to reasonable
    numerical error of torch kernel from baseline example.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA required for this test")

    dtype = torch.bfloat16

    # Create test inputs
    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    residual = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    weight = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    # Run baseline (torch compiled version)
    compiled_baseline = torch.compile(rms_norm_residual_block, fullgraph=True)
    baseline_output = compiled_baseline(x, residual, weight)

    # Run golden kernel
    golden_output = launch_fused_block(x, residual, weight)

    # Calculate differences
    max_diff = (baseline_output - golden_output).abs().max().item()
    mean_diff = (baseline_output - golden_output).abs().mean().item()

    print(f"  Output shape: {golden_output.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Verify outputs match (relaxed tolerance for bfloat16)
    torch.testing.assert_close(
        baseline_output,
        golden_output,
        atol=1e-1,  # Relaxed for bfloat16
        rtol=5e-2,
        msg=f"Golden kernel output does not match baseline (max_diff={max_diff:.6f})"
    )

    print(f"✓ Golden kernel test passed")


def test_golden_kernel_correctness_detailed():
    """
    Additional detailed correctness test with known values
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA required for this test")

    dtype = torch.bfloat16

    # Use smaller sizes for detailed testing
    small_seq_len = 16
    small_hidden_dim = 256

    # Create test inputs with known distribution
    torch.manual_seed(42)
    x = torch.randn(small_seq_len, small_hidden_dim, device=device, dtype=dtype)
    residual = torch.randn(small_seq_len, small_hidden_dim, device=device, dtype=dtype)
    weight = torch.ones(small_hidden_dim, device=device, dtype=dtype)  # Use ones for simplicity

    # Run baseline
    baseline_output = rms_norm_residual_block(x, residual, weight)

    # Run golden kernel
    golden_output = launch_fused_block(x, residual, weight)

    # Verify shape is correct
    assert golden_output.shape == (small_seq_len, small_hidden_dim // 2), \
        f"Expected shape {(small_seq_len, small_hidden_dim // 2)}, got {golden_output.shape}"

    # Verify outputs match (relaxed tolerance for bfloat16)
    torch.testing.assert_close(
        baseline_output,
        golden_output,
        atol=1e-1,
        rtol=5e-2,
        msg="Golden kernel detailed test failed"
    )

    # Verify no NaN or Inf values
    assert not torch.isnan(golden_output).any(), "Golden kernel produced NaN values"
    assert not torch.isinf(golden_output).any(), "Golden kernel produced Inf values"

    print(f"✓ Golden kernel detailed correctness test passed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {golden_output.shape}")
    print(f"  Output range: [{golden_output.min().item():.3f}, {golden_output.max().item():.3f}]")


def test_fusion_codegen():
    """
    Verify fusion status by inspecting generated Triton code.
    Current baseline: 2 kernels (unfused)
    Future with polyhedral fusion: 1 kernel (fused)
    """

    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(SEQ_LEN, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

    compiled_fn = torch.compile(rms_norm_residual_block, fullgraph=True)
    _, source_codes = run_and_get_code(compiled_fn, x, residual, weight)

    # Current baseline: 2 kernels (unfused)
    FileCheck().check_count("@triton.jit", 2, exactly=True).run(source_codes[0])
