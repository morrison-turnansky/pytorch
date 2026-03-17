# Polyhedral Fusion for RMSNorm + Chunking + Gating

## Problem: The Chunking Fusion Barrier

PyTorch Inductor generates **two separate kernels** for the RMSNorm + Chunking + Gating pattern common in LLMs:

```python
x = x + residual
variance = x.pow(2).mean(-1, keepdim=True)
x_normed = x * torch.rsqrt(variance + eps) * weight
gate, up = x_normed.chunk(2, dim=-1)  # ← FUSION BARRIER
return torch.nn.functional.silu(gate) * up
```

**The `.chunk()` operation is what torch.compile is NOT optimized for.**

### Inductor's 2-Kernel Approach

**Kernel 0 (Reduction):**
```python
# Load full row, compute variance, WRITE to HBM
tmp0 = load(x)                # LOAD 1
tmp1 = load(residual)         # LOAD 2
tmp2 = tmp0 + tmp1
variance = sum(tmp2 * tmp2) / N
store(variance)               # ← WRITE variance to HBM
```

**Kernel 1 (Pointwise):**
```python
# READ variance, RE-LOAD inputs
variance = load(variance)     # ← LOAD 3 (from HBM!)
tmp0 = load(x[0:N/2])        # LOAD 4 (gate)
tmp1 = load(residual[0:N/2]) # LOAD 5 (gate)
tmp2 = load(x[N/2:N])        # LOAD 6 (up)
tmp3 = load(residual[N/2:N]) # LOAD 7 (up)
# Apply normalization + chunking + gating
``

**Total memory traffic:**
- **7 loads** (2 full x+res + 2 half x+res + 1 variance)
- **2 stores** (variance + output)
- **Critical**: Variance roundtrip through HBM!

## Solution: Polyhedral-Style Fusion

Our golden kernel fuses both operations into **one kernel** that keeps variance in registers.

**Total memory traffic:**
- **6 loads** (1 full x+res + 2 half x+res)
- **1 store** (output only)
- **Critical**: No HBM roundtrip for variance!


## Performance Results (H200 GPU)

```
Inductor Baseline: 0.0796 ms/iter
Golden Fused:      0.0480 ms/iter

LATENCY REDUCTION: 0.0316 ms
SPEEDUP:           1.66x
```

## How Polyhedral Optimization Finds This

### 1. Dependence Analysis

Polyhedral compilers model statements and their dependencies:

```
S0: variance[i] = sum(x[i,j] + res[i,j])^2
S1: gate[i,j] = (x[i,j] + res[i,j]) * rsqrt(variance[i])
S2: up[i,j] = (x[i,k] + res[i,k]) * rsqrt(variance[i])
S3: out[i,j] = silu(gate[i,j]) * up[i,j]

Dependencies:
  S0 → S1 (RAW: variance)
  S0 → S2 (RAW: variance)
  S1 → S3, S2 → S3 (RAW: gate, up)
```

**Key insight:** No loop-carried dependencies → fusion is legal!

### 2. Fusion Profitability Analysis

```python
# Reuse analysis
Temporal reuse of variance: N accesses per scalar
Spatial locality: variance[i] accessed repeatedly in loop over j

# Distance vector
RAW(S0 → S1): variance written once, read N/2 times
RAW(S0 → S2): variance written once, read N/2 times

Conclusion: Fusing saves N memory accesses!
```

### 3. Register Promotion

Polyhedral technique: **Scalar expansion + contraction**

```
Original:  variance[i] stored in array (HBM)
Optimized: variance promoted to register-resident scalar

Transformation:
- Expand: Create scalar tmp7 per thread block
- Use: Replace variance[i] reads with tmp7
- Contract: Never materialize variance[] array
```

This is exactly what `tmp7` does in our golden kernel!

### 4. Schedule Transformation

```python
# Original (2 kernels)
for i in range(M):
    variance[i] = reduce(...)        # Kernel 0
for i in range(M):
    for j in range(N/2):
        out[i,j] = compute(variance[i])  # Kernel 1

# Transformed (1 kernel)
for i in range(M):  # ← Fuse outer loops
    var_reg = reduce(...)  # Keep in register
    for j in range(N/2):
        out[i,j] = compute(var_reg)  # Use from register
```

## Why Inductor Misses This

1. **Chunking barrier**: `.chunk()` creates separate views, seen as independent consumers
