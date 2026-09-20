# `cfg(cuda_arch*)`: architecture-conditional device code

`cargo oxide` resolves one target architecture per build and hands it to rustc
as ordinary `--cfg` flags, so device code picks an implementation the same way
portable Rust picks one per platform — with `#[cfg]`, not with a runtime check.

```bash
cargo oxide build arch_cfg --arch sm_75
crates/rustc-codegen-cuda/examples/arch_cfg/verify-code-shape.sh

cargo oxide build arch_cfg --arch sm_86
crates/rustc-codegen-cuda/examples/arch_cfg/verify-code-shape.sh

cargo oxide run arch_cfg
```

The kernel `warp_sum` reduces a warp two ways:

| Build | `cuda_arch_min = "80"` | Generated |
|:--|:--|:--|
| `--arch sm_86` | set | one `redux.sync.add.s32` |
| `--arch sm_75` | unset | five `shfl.sync.bfly` + add |

`redux.sync` does not exist below sm_80, so on Turing the first arm is not
merely slower — it cannot be lowered at all. rustc evaluates `#[cfg]` before
MIR exists, so the unselected arm is never collected and never reaches the
codegen backend. `verify-code-shape.sh` proves that for the sm_75 build by
checking that `arch_cfg.ll` does not so much as mention `llvm.nvvm.redux`; a
backend that merely dead-stripped the call later would still leave it there.

There is no `ctx.compute_capability()` check in this example. The kernel
instead writes `cfg!(cuda_arch_min = "80")` into the last output slot, so the
host reports the arm that was actually compiled rather than the one it assumes
from the target it requested. The final line is:

```text
arch_cfg: PASS (redux path)
```

or, for an sm_75 build:

```text
arch_cfg: PASS (shuffle fallback)
```

GPU-less CI builds both architectures and runs the shape check after each,
through `scripts/smoketest.sh --compile-only`. To run that locally:

```bash
scripts/smoketest.sh --compile-only arch_cfg
```

Building with no architecture at all is the third interesting case: no
`cuda_arch*` cfgs are set, so every `#[cfg(cuda_arch...)]` takes its fallback
arm and `cargo oxide` warns that it did. In a crate that declares nothing,
rustc *also* reports each use as an unexpected cfg condition name — a second,
louder signal that the build is unpinned. Opt into a hard error with:

```toml
[lints.rust]
unexpected_cfgs = "deny"
```

This example does not get that second signal, and its `Cargo.toml` explains
why: CI lints every example with a plain `cargo clippy -- -D warnings`, which
runs through no wrapper and so sees an undeclared name, so the example declares
`cuda_arch_min` itself. Declaring a name silences it for *every* build,
unpinned ones included. A kernel crate that only ever builds through
`cargo oxide` should declare nothing and keep the warning.
