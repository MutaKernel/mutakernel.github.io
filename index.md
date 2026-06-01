# MutaKernel: Validating the Validators of LLM-Generated GPU Kernels

> Anonymous supplementary material, released to preserve double-blind review.

## Abstract

Large language models are increasingly used to automate performance optimization in
computer systems. In AI-Driven Research for Systems (ADRS), LLMs generate candidate
optimizations, evaluate them on target workloads, and refine later candidates using
measured feedback. Although this loop depends critically on evaluator quality, prior
ADRS work has largely overlooked this issue and focused mainly on performance
feedback. As a result, correctness validation remains underexamined, creating a risk
of reward hacking where generated code passes limited validation while violating the
intended semantics.

This paper studies evaluator reliability in LLM-driven GPU kernel optimization, a
rapidly growing ADRS sub-domain. We develop a mutation-testing methodology, supported
by a four-layer equivalence-filtering pipeline and equivalent mutant detection
approach, to measure validator fault-detection effectiveness. Our study finds that
168 of the 534 kernels accepted by KernelBench are buggy. Guided by these findings,
we propose MutaKernel, an enhanced validator that strengthens testing along five
orthogonal dimensions and allocates testing budget by fault category. MutaKernel
detects 222 defective kernels among 597 public kernels released by four prior tools
and previously accepted by the KernelBench validator. These results show that
validator reliability is a key bottleneck for trustworthy ADRS, and that targeted
test strengthening can substantially reduce the risk of accepting incorrect optimized
kernels.

## Workflow

<figure>
  <img src="../figures/study-workflow.png" alt="Three-stage study workflow" width="100%">
  <figcaption><strong>Figure 1.</strong> The three-stage study workflow: mutant
  generation, equivalence filtering, and validator evaluation.</figcaption>
</figure>

<figure>
  <img src="../figures/mutakernel-workflow.png" alt="MutaKernel workflow" width="100%">
  <figcaption><strong>Figure 2.</strong> MutaKernel workflow. Survived and
  candidate-equivalent mutants from EMD enter tier classification, which assigns
  testing intensity. Five test dimensions then run across a main track and a
  config-stress track. Final labeling produces a per-mutant verdict.</figcaption>
</figure>

---

# Supplementary Material

- [A. Mutation operators](#a-mutation-operators)
- [B. EMD Layer-1 static rules](#b-emd-layer-1-static-rules)
- [C. Stress-policy library](#c-stress-policy-library)
- [D. ADRS repair experiment (RQ5)](#d-adrs-repair-experiment-rq5)

---

## A. Mutation operators

*Per-operator details for the 16 operators in Table 1.*

| Category | Operators |
|----------|-----------|
| A — Classical Arithmetic | `arith_replace`, `relop_replace`, `const_perturb` |
| B — GPU Parallel Semantics | `index_replace`, `sync_remove`, `mask_boundary`, `launch_config_mutate` |
| C — ML Numerical Semantics | `stab_remove`, `epsilon_modify`, `scale_modify`, `init_modify`, `acc_downgrade`, `cast_remove`, `reduction_reorder` |
| D — LLM-Specific Error Patterns | `broadcast_unsafe`, `layout_assume` |

For each operator: the sites it matches, the mutation it applies, and the escape
mechanism (why the defect can pass a default-input `allclose` check).

### A — Classical Arithmetic

**`arith_replace`**
- *Sites.* Binary `+ - * /` in the Python/Triton AST (located via `tokenize`,
  excluding decorator expressions) and in the CUDA C++ string (character scan that
  excludes compound operators `++ += -- -= -> ** *= // /= /*`, requires a binary left
  operand, and filters pointer declarations against a 37-entry C-type keyword list).
- *Mutation.* One-way per site: `+→-`, `-→+`, `*→/`, `/→*`.
- *Escape.* On small-magnitude inputs the arithmetic difference falls below the `1e-2` tolerance.

**`relop_replace`**
- *Sites.* `Compare` nodes in Python/Triton; in CUDA, two-character ops
  (`<= >= == !=`) preferred, with shifts (`<< >>`), kernel-launch syntax (`<<< >>>`),
  and template angle brackets filtered out.
- *Mutation.* `< ↔ <=`, `> ↔ >=`, `== ↔ !=`.
- *Escape.* Random inputs rarely land on the boundary that distinguishes `<` from `<=`.

**`const_perturb`**
- *Sites.* Integer literals (excluding `0`/`1` and GPU-scheduling configuration lines
  containing `BLOCK_SIZE`, `WARP_SIZE`, `num_warps`, `threadIdx`, `blockDim`, `dim3`,
  etc.) and float literals (excluding `0.0`/`1.0`; handles scientific notation and
  `f/F/l/L` suffixes).
- *Mutation (two per site).* Integers → `+1` and `-1`; floats → `×1.01` and `×0.99`.
- *Escape.* Sub-1% perturbations are absorbed by the tolerance.

### B — GPU Parallel Semantics

**`index_replace`**
- *Sites.* Triton `tl.program_id(N)`; CUDA `threadIdx|blockIdx|blockDim|gridDim . {x,y,z}`.
- *Mutation (two per site).* Swap to each of the other two axes.
- *Escape.* Under a 1-D launch the other axes degenerate to the same value, so
  symmetric inputs leave the swap unobservable.

**`sync_remove`**
- *Sites.* `tl.debug_barrier()` / `__syncthreads()`.
- *Mutation.* Delete the barrier.
- *Equivalence guard.* Flagged as a reduction-tail sync (algorithmically absorbed)
  when (1) the first non-trivial statement within four lines is a `threadIdx.x == 0` /
  `tid == 0` guard, or (2) it is the last `__syncthreads()` and all subsequent
  shared-memory accesses use literal index `[0]`.
- *Escape.* At small magnitudes the race produces bit jitter within the loose tolerance.

**`mask_boundary`**
- *Sites.* Triton lines with `mask=` / `tl.where`; CUDA boundary guards involving
  `threadIdx`/`blockIdx` or early-return comparisons (`if (i >= n) return;`).
- *Mutation.* Tightening only: `x < y → x < y-1`; on the `>=` path, `x >= y → x > y`
  and `x >= y → x >= y+1`. The loosening direction is not emitted (a loosened mask
  almost always reads out of bounds → stillborn).
- *Escape.* The excluded elements lie in the padding region, never reached under
  default inputs.

**`launch_config_mutate`**
- *Sites.* `triton.cdiv(...)`, `X // Y`, CUDA `cdiv(...)`, and the
  `(N + BLOCK - 1) / BLOCK` ceiling-division idiom.
- *Mutation.* `expr → expr - 1` only (the `+1` direction only launches redundant
  threads the in-kernel mask filters, yielding an equivalent mutant).
- *Escape.* Grid-stride loops absorb the missing block on divisible shapes; the gap
  appears only on non-divisible batch sizes.

### C — ML Numerical Semantics

**`stab_remove`**
- *Target.* The `x - max(x)` trick in softmax.
- *Sites.* Triton/PyTorch `var - tl.max(var, …)` / `torch.max` / `var.max(…)`
  (verified that the subtracted term equals the variable); CUDA `expf(val - row_max)`
  and the split form `val - max_val`.
- *Mutation.* Drop the `- max(...)` term.
- *Escape.* For small-magnitude inputs `exp(x)` does not overflow, so both versions agree.

**`epsilon_modify`**
- *Target.* Stability epsilons in LayerNorm / safe-division / log.
- *Sites.* Scientific notation (`1e-5`, `1e-6f`, …) and decimals with ≥4 leading zeros
  (`0.00001f`), filtered to magnitude ≤ `1e-3`, in both Python and CUDA.
- *Mutation (two per site).* `eps → 0` and `eps → 1e-2` (suffix preserved).
- *Escape.* Most random inputs never drive the denominator near zero.

**`scale_modify`**
- *Target.* The `1/√d` scaling in attention / normalization.
- *Sites.* `1 / sqrt(x)` / `1.0 / torch.sqrt(x)` (drop the `1/` prefix); the `rsqrt(x)`
  family (extract the argument, e.g. `rsqrt(var+eps) → var+eps`); CUDA `1.0f / sqrtf(x)`
  and `rsqrtf(expr)`.
- *Mutation.* Invert or remove the inverse-square-root scaling.
- *Escape.* When `d` is small the scale difference can stay within tolerance.

**`init_modify`**
- *Target.* Initial values of min/max reductions.
- *Sites.* Python `float('-inf')`/`float('inf')`/`tl.full(…, -inf, …)`; CUDA
  `±INFINITY`, `±FLT_MAX`, `±HUGE_VALF`.
- *Mutation (two per site).* `±inf → 0.0` and `±inf → ±1e10`.
- *Escape.* If the data range lies within `[-1e10, 1e10]`, `-1e10` behaves like `-inf`.

**`acc_downgrade`**
- *Target.* FP32 accumulator demoted to FP16.
- *Sites.* `tl.zeros(…, dtype=float32)`, `torch.zeros(…, dtype=float32)`,
  `.to(float32)`, `.float()`, `.double()`; CUDA `static_cast<float>`, `(float)expr`,
  `__half2float`.
- *Mutation.* Demote one precision level (`float32→float16`, `float→__half`,
  `double→float`; strip `__half2float`).
- *Escape.* On small reductions FP16 precision suffices; the gap appears only at scale.

**`cast_remove`**
- *Sites.* `.to(float32/float16/bfloat16/int32/int64)`, `.float()`, `.half()`,
  `tl.cast(x, dtype)`; CUDA `static_cast<T>(expr)`, `(float)expr`,
  `__float2half`/`__half2float`.
- *Mutation.* Strip the cast wrapper.
- *Equivalence guard.* A `static_cast<T>` is flagged redundant when the LHS type
  already matches `T` (C++ implicit conversion makes the cast a no-op).
- *Escape.* When the input dtype already matches the target, removing the cast has no effect.

**`reduction_reorder`**
- *Target.* Non-associativity of floating-point addition.
- *Sites.* Triton/PyTorch `tl.sum(t, axis=N)` → `tl.sum(t[::-1], axis=N)`,
  `torch.sum(t, dim=N)` → `torch.sum(torch.flip(t,(N,)), dim=N)`. (CUDA hand-written
  reductions are out of scope for this operator.)
- *Mutation.* Reverse the summation order.
- *Escape.* The difference is typically at the `1e-7` level, far below `1e-2`.

### D — LLM-Specific Error Patterns

These act on the Python layer (PyTorch tensor ops in the wrapper).

**`broadcast_unsafe`**
- *Sites.* `.expand(...)`, `.expand_as(...)`, `.broadcast_to(...)`,
  `tl.broadcast_to(...)`, `.unsqueeze(...)`. (Not `.reshape()`/`.view()`: removing a
  required reshape almost certainly triggers a shape mismatch → stillborn.)
- *Mutation.* Delete the alignment call (or extract the first argument for `tl.broadcast_to`).
- *Escape.* Symmetric values mask a wrong broadcast direction.

**`layout_assume`**
- *Sites.* `.contiguous()` / `.is_contiguous()`.
- *Equivalence guard.* A `.contiguous()` is mutated only when preceded on the same
  line by a layout-changing op (`.transpose`, `.permute`, `.mT`, `.T`, `.t()`, or a
  slice); a purely defensive `.contiguous()` is skipped as equivalent.
- *Mutation.* Delete the call.
- *Escape.* A contiguous input layout happens to satisfy the assumption.

---

## B. EMD Layer-1 static rules

*Rule definitions for Layer 1 of the four-layer EMD pipeline.*

Layer 1 applies four operator-aware static rules, adapted from EMS (Kushigian et al.,
ISSTA 2024) to the CUDA execution model. A rule hit promotes a survived mutant to
**strict-equivalent** without executing the kernel. Rules are tried in order; any
exception during matching is caught and the rule is skipped (conservative).

**Rule 1 — `boundary_unreachable`**
- *Applies to.* `relop_replace`, `mask_boundary`.
- *Principle.* In CUDA, `threadIdx.d ∈ [0, blockDim.d − 1]`. Therefore
  `(t < B) ≡ (t <= B) ≡ True` and `(t > B) ≡ (t >= B) ≡ False` for every legal thread,
  so swapping `<` for `<=` (or `>` for `>=`) on a `threadIdx.d` vs `blockDim.d`
  comparison does not change the guard's truth value.
- *Match.* Regex `threadIdx.([xyz]) (<|<=|>|>=) blockDim.([xyz])`; confirm the two
  dimension identifiers are identical and the operator pair is `{<, <=}` or `{>, >=}`
  between original and mutant.

**Rule 2 — `dead_write`**
- *Applies to.* `arith_replace`, `const_perturb`, `scale_modify`, `init_modify`.
- *Principle.* If a mutation changes the right-hand side of an assignment but the
  variable is unconditionally reassigned before its next read, the mutation is
  overwritten and cannot affect later computation.
- *Match.* Identify the assigned variable (excluding CUDA keywords such as `if`,
  `for`, `__global__`); scan forward from the mutated line. If a reassignment is
  reached before any read, return equivalent; if a read comes first, return
  non-equivalent. Blank and comment lines are skipped.

**Rule 3 — `mask_noreach`**
- *Applies to.* `mask_boundary`.
- *Principle.* Tightening a boundary guard (`idx < n → idx < n−1`) affects only
  padding-region threads, whose writes fall outside the valid output range.
- *Match.* Regex `(idx|tid|index|i|gid) (<|<=) (\w+)`; recognize tightening (`<=` → `<`,
  or the right-hand bound decremented).

**Rule 4 — `dead_host_constant`**
- *Applies to.* `const_perturb`.
- *Principle.* Under the fixed-shape contract, `get_inputs()` and `get_init_inputs()`
  always come from the reference module, not the mutant. So if `const_perturb` changes
  a module-level constant (e.g. `N = 2048 → 2049`) that is not referenced by the
  `ModelNew`/`Model` class, the mutation is dead code that can never affect the
  kernel's execution path.
- *Implementation (conservative short-circuit order).*
  (1) operator must be `const_perturb`;
  (2) the file must parse;
  (3) the mutated line must not lie inside any `ClassDef`;
  (4) nor inside any `FunctionDef` / `AsyncFunctionDef`;
  (5) it must be a simple module-level assignment with a single `Name` target;
  (6) if the assigned name appears anywhere (Store or Load) inside `ModelNew` or
  `Model`, return non-equivalent;
  (7) otherwise the mutant is strict-equivalent.
  Step 6 is an existence check by design: it conservatively refuses to declare dead
  code whenever a same-named variable is present, even if unrelated.

```
Layer1_StaticRules(m):
    rules ← [boundary_unreachable, dead_write, mask_noreach, dead_host_constant]
    for rule in rules:
        try:
            if rule(m): return (STRICT, rule.name)
        except Exception:
            continue                  # conservative: skip on error
    return (CONTINUE, None)
```

---

## C. Stress-policy library

*The per-family policy list and the operator-to-policy priority map.*

The 21 stress policies are used by the `value_stress` and `training_stress`
dimensions. Every policy keeps shape and dtype fixed and changes only the numeric
values in the tensors.

### Per-family policy list

**Family 1 — extreme numeric distributions**

| Policy | Numeric character | Targeted behavior | Source |
|--------|-------------------|-------------------|--------|
| `large_magnitude` | `randn × 1000` | arithmetic-overflow path divergence | FPGen [ICSE'20] |
| `extreme_magnitude` | `randn × 1e6` | more aggressive overflow exposure | FPGen [ICSE'20] |
| `near_overflow` | `randn ×` dtype upper bound (FP32 ~1e30, FP16 ~60000) | path divergence near the precision limit | FPGen [ICSE'20] |
| `near_zero` | `randn × 1e-7` | epsilon-related branches (÷0, rsqrt) | FPGen [ICSE'20] |
| `denormals` | `randn × 1e-38` | denormal-number handling differences | FPGen [ICSE'20] |
| `near_epsilon` | uniform in `[1e-7, 1e-5]` | epsilon decision boundary | FPGen [ICSE'20] |
| `mixed_extremes` | 50% × 10000 + 50% × 0.0001 | mixed extremes expose precision paths | Laguna [SC'24] |

**Family 2 — boundary values and sign distributions**

| Policy | Numeric character | Targeted behavior | Source |
|--------|-------------------|-------------------|--------|
| `all_negative` | `−|randn| × 100` | exposes `init_modify` (0 → positive) masking | BVA |
| `all_positive` | `|randn| × 100` | all-positive control | BVA |
| `boundary_last_element` | `randn` with last element = 1e4 | off-by-one boundary (`mask_boundary`) | BVA |
| `relop_boundary_hit` | `arange % 10` (integer-valued) | relational decision boundary (`<` vs `<=`) | BVA |

**Family 3 — sparsity gradient**

| Policy | Zero fraction | Targeted behavior | Source |
|--------|---------------|-------------------|--------|
| `dense_nonzero` | `|randn| + 1.0` (0% zeros) | remove zero-masking of arithmetic differences | pilot |
| `sparse` | 90% zeros + 10% `randn × 100` | ordinary sparse activations | BVA |
| `sparse_extreme` | 99% zeros + 1% × 1e4 | extreme sparsity boundary | BVA |

**Family 4 — structured / position-sensitive**

| Policy | Numeric character | Targeted behavior | Source |
|--------|-------------------|-------------------|--------|
| `structured_ramp` | `[0, 1/n, 2/n, …]` | distinguishable positions; exposes index degeneration | Mu2 [ISSTA'23] |
| `head_heavy` | first 25% extreme, rest near-zero | exposes index degeneration processing only the head | Mu2 [ISSTA'23] |
| `tail_heavy` | last 25% extreme, rest near-zero | exposes index degeneration skipping the tail | Mu2 [ISSTA'23] |

**Family 5 — reduction/accumulation adversarial + special behavior**

| Policy | Numeric character | Targeted behavior | Source |
|--------|-------------------|-------------------|--------|
| `alternating_sign` | alternating ± × 100 | summation-order sensitivity (`reduction_reorder`) | Higham [2002] |
| `reduction_adversarial` | alternating ±1e4 + small noise | maximizes FP reduction error | Higham [2002] |
| `uniform_constant` | all 88.0 | exposes shift-invariance assumptions (`scale_modify`) | Magneto [ISSTA'21] |
| `init_sensitive` | random all-positive or all-negative | exposes min/max initialization differences | pilot |

### Operator-to-policy priority map

Orders the policies for `value_stress` and `training_stress` so the policy most likely
to kill a given operator runs first.

| Operator | Recommended policies (in order) | Survival mechanism |
|----------|---------------------------------|--------------------|
| `epsilon_modify` | `near_zero`, `denormals`, `dense_nonzero` | eval-mode `var ≈ 1` masks the eps difference |
| `scale_modify` | `uniform_constant`, `structured_ramp` | `running_var = 1.0` cancels the scale difference |
| `stab_remove` | `large_magnitude`, `near_overflow` | `randn ∈ [−3, 3]` never triggers the overflow path |
| `cast_remove` | `near_overflow`, `large_magnitude` | under FP32, `static_cast` is identity |
| `init_modify` | `all_negative`, `sparse` | positive `randn` masks `min_val = 0` initialization |
| `acc_downgrade` | `mixed_extremes`, `large_magnitude` | small-value accumulation hides precision loss |
| `reduction_reorder` | `mixed_extremes`, `alternating_sign` | random ± values cancel, hiding order differences |
| `broadcast_unsafe` | `structured_ramp` | symmetric values mask the wrong broadcast direction |
| `layout_assume` | `structured_ramp` | contiguous layout happens to satisfy the assumption |
| `index_replace` | `structured_ramp`, `large_magnitude`, `head_heavy`, `tail_heavy` | 1-D launch degenerates `blockIdx.y/z` to 0 |
| `mask_boundary` | `boundary_last_element`, `sparse`, `sparse_extreme` | default inputs never reach the boundary |
| `sync_remove` | `large_magnitude`, `mixed_extremes` | race differences absorbed by tolerance at small magnitudes |
| `launch_config_mutate` | `structured_ramp`, `large_magnitude` | grid-stride loops absorb the config difference |
| `arith_replace` | `large_magnitude`, `mixed_extremes`, `dense_nonzero` | `+/−/×/÷` differences invisible at small magnitudes |
| `relop_replace` | `boundary_last_element`, `structured_ramp`, `sparse_extreme` | `randn` values do not land on the decision boundary |
| `const_perturb` | `near_zero`, `large_magnitude` | perturbation absorbed by `allclose` tolerance |

```python
STRATEGY_MAP = {
    "epsilon_modify":       ["near_zero", "denormals", "dense_nonzero"],
    "scale_modify":         ["uniform_constant", "structured_ramp"],
    "stab_remove":          ["large_magnitude", "near_overflow"],
    "cast_remove":          ["near_overflow", "large_magnitude"],
    "init_modify":          ["all_negative", "sparse"],
    "acc_downgrade":        ["mixed_extremes", "large_magnitude"],
    "reduction_reorder":    ["mixed_extremes", "alternating_sign"],
    "broadcast_unsafe":     ["structured_ramp"],
    "layout_assume":        ["structured_ramp"],
    "index_replace":        ["structured_ramp", "large_magnitude", "head_heavy", "tail_heavy"],
    "mask_boundary":        ["boundary_last_element", "sparse", "sparse_extreme"],
    "sync_remove":          ["large_magnitude", "mixed_extremes"],
    "launch_config_mutate": ["structured_ramp", "large_magnitude"],
    "arith_replace":        ["large_magnitude", "mixed_extremes", "dense_nonzero"],
    "relop_replace":        ["boundary_last_element", "structured_ramp", "sparse_extreme"],
    "const_perturb":        ["near_zero", "large_magnitude"],
}
```

---

## D. ADRS repair experiment (RQ5)

RQ5 asks whether the defects MutaKernel exposes are real bugs an advanced LLM can fix,
or testing artifacts no model can act on.

### Setup

We run Claude Opus 4.5 on two defect sets.

**Repair-KB** contains 18 KernelBench validator-accepted kernels for which the
original LLM-generated CUDA kernel passes the baseline `torch.allclose` check at
`atol = rtol = 1e-2` but fails at least one of MutaKernel's five stress dimensions on
the same fixed-shape contract. These 18 kernels are surfaced by running MutaKernel's
stress pipeline on each of the 90 validator-accepted kernels from the study, as a
byproduct of the mutation experiments, alongside the 166 additional mutant kills
reported in RQ3.

**Repair-Agent** contains the 104 CUDA-Agent kernels that MutaKernel's five stress
dimensions flag as defective in RQ4 (the `|S|` set on CUDA-Agent, of which 101 are
stress-only defects `|S \ B|` and 3 overlap with the baseline positives).

Each kernel gets up to three repair rounds. Each round receives the kernel source, the
PyTorch reference, and the previous round's per-dimension failure details, and emits a
new kernel that MutaKernel re-tests. A kernel is **framework-fixed** when some round
passes every stress dimension.

### What counts as a real fix

Framework-fixed alone is not enough. An LLM can pass MutaKernel by deleting the CUDA
kernel and calling `torch.matmul` or `nn.Conv2d` on the reference path. We therefore
audit every framework-fixed kernel and label it a **real CUDA fix** only when the
repaired source keeps a custom `__global__` kernel and `forward` actually calls it.
Repairs that fail this audit fall into three classes:

- **PyTorch fallback** — replaces the kernel with `torch.*` or `nn.*`.
- **Library wrapper** — keeps a `load_inline` shell but its body calls `cublasSgemm`
  or `torch::mm` with no `__global__` definitions.
- **Non-substantive** — changes only a compiler flag, or repairs a kernel whose
  Round 0 re-test finds no reproducible bug.

### Result

Both settings reach framework-fixed rates above 86%, but both real-fix rates fall far
below that. Repair-KB drops from 88.9% framework-fixed to 38.9% real-fix; Repair-Agent
drops from 86.5% to 14.4%. PyTorch fallback dominates Repair-Agent, where 72 of the 90
framework-fixes delete the CUDA kernel entirely and route through `nn.*` or `torch.*`.
Repair-KB shows the same pattern at smaller scale, plus 4 library-wrapper repairs
specific to its GEMM-heavy targets.

Framework-fixed counts kernels that pass MutaKernel in some round. Real-fix
additionally requires that `forward` call a custom `__global__` kernel.

| Outcome | Repair-KB (18) | Repair-Agent (104) |
|---------|---------------:|-------------------:|
| Framework-fixed | 16 (88.9%) | 90 (86.5%) |
| &nbsp;&nbsp;Real CUDA fix | **7 (38.9%)** | **15 (14.4%)** |
| &nbsp;&nbsp;PyTorch fallback | 3 (16.7%) | 72 (69.2%) |
| &nbsp;&nbsp;Library wrapper | 4 (22.2%) | 0 |
| &nbsp;&nbsp;Non-substantive | 2 (11.1%) | 3 (2.9%) |
| Unfixed | 2 (11.1%) | 14 (13.5%) |

### Extra rounds buy fallback, not fixes

Round 1 produces all 15 Repair-Agent real fixes. Rounds 2 and 3 add 26 more
framework-fixes and 0 real fixes; every late-round fix is a PyTorch fallback or a
non-substantive change. The model does its real CUDA repair attempt in Round 1; later
rounds just push it to abandon CUDA more aggressively to clear the stress test.

| Round | Framework-fixed | Real CUDA fix |
|-------|----------------:|--------------:|
| Round 1 | 64 | 15 |
| Round 2 | 16 | 0 |
| Round 3 | 10 | 0 |
| Total | 90 | 15 |

### L3 kernels get zero real fixes

Repair-Agent's 15 real fixes split 14 on L2, 1 on L1, and 0 on L3. The 20 L3 kernels
reach a framework-fixed rate of 95% (19/20), but every L3 framework-fix abandons
CUDA-level repair: 17 fall back to PyTorch and 2 are non-substantive TF32-only changes.
L3 defects are the hardest to repair at the CUDA level, and the model gives up on all
of them.

### Real fixes are often slower

We benchmark the 7 Repair-KB real fixes against the original buggy kernels under 20
warmup and 50 timed runs. Four real fixes run 1.4 to 3.9 times slower than the
original; in these cases the original's speed came from cutting a corner the real fix
has to put back, like a truncated matmul loop. Two real fixes run 2.4 and 6.6 times
faster, because they removed a redundant computation while restoring correctness.
KernelBench's default-input validator cannot tell these apart, so its best-of-N
selection rewards fast-but-buggy kernels.

### Finding

The defects MutaKernel exposes are real and hard. Under strict audit, an advanced LLM
can repair only 38.9% of them on KernelBench and 14.4% on CUDA-Agent, against
framework-reported rates above 86% on both. This rules out the alternative that
MutaKernel's gains over the baseline are testing artifacts. Combined with RQ3's
independent audit confirming the residual mutants are equivalents rather than missed
bugs, the gap between the baseline and MutaKernel measured in RQ3 reflects real
defects.
