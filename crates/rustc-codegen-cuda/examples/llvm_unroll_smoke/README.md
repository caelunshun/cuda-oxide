# `#[llvm_unroll]`: unrolling by LLVM

`#[unroll]` is handled by cuda-oxide's own MIR pass, which recognizes explicit
counted `while` loops and warns when it cannot prove a trip count.
`#[llvm_unroll]` instead records `!llvm.loop` metadata for LLVM's unroller,
which runs as part of `opt -O2` and derives trip counts with SCEV.

```bash
cargo oxide run llvm_unroll_smoke
```

The kernels use range-based `for` loops on purpose: that is a shape cuda-oxide's
own analysis does not recognize, so `#[unroll]` would warn and change nothing
there, while LLVM handles it without difficulty.

| Kernel | Annotation | Expected generated shape |
|:--|:--|:--|
| `full` | `#[llvm_unroll]` | Fully unrolled; no loop remains |
| `partial` | `#[llvm_unroll(4)]` | Four body copies per trip, plus a remainder |
| `control` | none | Whatever LLVM picks on its own |

`control` is the honest baseline. LLVM unrolls this loop at `-O2` even without
an annotation — it just chooses the factor itself. `#[llvm_unroll(N)]` is how
you pick the factor instead of accepting that default, and the shape check
proves the factor in the output came from the annotation by requiring `partial`
and `control` to differ.

Each iteration writes its accumulator with a volatile store, which optimization
cannot merge away, so body copies are countable in the generated LLVM IR.

GPU-less CI runs the example's `verify-code-shape.sh` automatically through
`scripts/smoketest.sh --compile-only`. To run that check locally:

```bash
scripts/smoketest.sh --compile-only llvm_unroll_smoke
```

The request is inert in builds that skip `opt` (`CUDA_OXIDE_NO_OPT=1` and full
variable-debug builds); those builds warn rather than silently drop it.
