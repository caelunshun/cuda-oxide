/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The `cuda_arch*` conditional-compilation vocabulary.
//!
//! Device code asks which GPU architecture it is being compiled for with
//! plain `#[cfg]`. `cargo oxide` derives the answer from the resolved target
//! and injects it as global `--cfg` rustflags, so `#[cfg]` is evaluated by
//! rustc before MIR exists and the unselected arm is never collected.
//!
//! This module is the single owner of both halves of that contract: the cfg
//! names and values a target produces ([`arch_cfgs`]) and the universe rustc
//! is told to expect ([`check_cfg_specs`]). Keeping them together is what
//! makes a typo like `cuda_arch_min = "8O"` a warning rather than a silently
//! false predicate.
//!
//! ```text
//! sm_90a --> cuda_arch = "90"
//!            cuda_arch_min = "70" .. "90"     (every recorded capability <= 90)
//!            cuda_arch_target = "sm_90a"
//!            cuda_arch_specific = "90"        ('a' suffix)
//!            cuda_arch_family_specific = "90" ('a' or 'f' suffix)
//! ```
//!
//! The mapping mirrors CUDA C++: `cuda_arch` is `__CUDA_ARCH__ == N`,
//! `cuda_arch_min` is `__CUDA_ARCH__ >= N`, and the two `specific` names are
//! `__CUDA_ARCH_SPECIFIC__` / `__CUDA_ARCH_FAMILY_SPECIFIC__`.
//!
//! Ordering is numeric, exactly as in CUDA, so `sm_120` satisfies
//! `cuda_arch_min = "100"` even though consumer Blackwell is not a superset
//! of datacenter Blackwell. Datacenter-only features must gate on
//! [`CFG_ARCH_SPECIFIC`], [`CFG_ARCH_FAMILY_SPECIFIC`] or [`CFG_ARCH_TARGET`]
//! instead.

use crate::{CudaArch, RECORDED_PTX_FLOORS};

/// Exact compute capability of the target being compiled for.
///
/// CUDA C++ analogue: `__CUDA_ARCH__ == N`.
pub const CFG_ARCH: &str = "cuda_arch";

/// Every recorded compute capability the target is at least as new as.
///
/// Emitted once per distinct capability, the same shape rustc itself uses for
/// `target_has_atomic`, so `#[cfg(cuda_arch_min = "80")]` is one predicate
/// rather than a hand-written range.
///
/// CUDA C++ analogue: `__CUDA_ARCH__ >= N`.
pub const CFG_ARCH_MIN: &str = "cuda_arch_min";

/// The exact target spelling, normalized to `sm_XX[a|f]`.
///
/// This is the only name that distinguishes `sm_100a` from `sm_100f` from
/// `sm_100`, so it is the escape hatch when numeric ordering is the wrong
/// question.
pub const CFG_ARCH_TARGET: &str = "cuda_arch_target";

/// Set when the target carries the architecture-specific `a` suffix.
///
/// CUDA C++ analogue: `__CUDA_ARCH_SPECIFIC__`.
pub const CFG_ARCH_SPECIFIC: &str = "cuda_arch_specific";

/// Set when the target carries an `a` or `f` suffix.
///
/// CUDA C++ analogue: `__CUDA_ARCH_FAMILY_SPECIFIC__`.
pub const CFG_ARCH_FAMILY_SPECIFIC: &str = "cuda_arch_family_specific";

/// Every cfg name this module owns.
///
/// `cargo oxide` strips inherited copies of exactly these names before adding
/// its own, so a stale `RUSTFLAGS='--cfg cuda_arch="70"'` in the environment
/// cannot make device code believe it is building for another GPU.
pub const ALL_CFG_NAMES: &[&str] = &[
    CFG_ARCH,
    CFG_ARCH_MIN,
    CFG_ARCH_TARGET,
    CFG_ARCH_SPECIFIC,
    CFG_ARCH_FAMILY_SPECIFIC,
];

/// One `name="value"` conditional-compilation setting.
///
/// Every name is key/value rather than a bare flag so that `--check-cfg` can
/// enumerate the legal values and rustc can reject a misspelled one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CfgPair {
    /// One of [`ALL_CFG_NAMES`].
    pub name: &'static str,
    /// The value, unquoted.
    pub value: String,
}

impl CfgPair {
    /// Render as the `--cfg` argument rustc expects: `name="value"`.
    pub fn render(&self) -> String {
        format!("{}=\"{}\"", self.name, self.value)
    }
}

/// Distinct recorded compute capabilities, ascending.
fn recorded_capabilities() -> Vec<u32> {
    let mut capabilities: Vec<u32> = RECORDED_PTX_FLOORS
        .iter()
        .map(|entry| entry.capability)
        .collect();
    capabilities.sort_unstable();
    capabilities.dedup();
    capabilities
}

/// Distinct recorded capabilities that have an entry with one of `suffixes`.
fn recorded_capabilities_with_suffix(suffixes: &[char]) -> Vec<u32> {
    let mut capabilities: Vec<u32> = RECORDED_PTX_FLOORS
        .iter()
        .filter(|entry| {
            entry
                .suffix
                .is_some_and(|suffix| suffixes.contains(&suffix))
        })
        .map(|entry| entry.capability)
        .collect();
    capabilities.sort_unstable();
    capabilities.dedup();
    capabilities
}

/// Derive the full `cuda_arch*` cfg set for a resolved target.
///
/// The `min` values come from [`RECORDED_PTX_FLOORS`] rather than from every
/// integer below the capability, so the emitted set is always a subset of the
/// universe [`check_cfg_specs`] declares.
pub fn arch_cfgs(arch: &CudaArch) -> Vec<CfgPair> {
    let capability = arch.capability();
    let mut cfgs = vec![CfgPair {
        name: CFG_ARCH,
        value: capability.to_string(),
    }];
    cfgs.extend(
        recorded_capabilities()
            .into_iter()
            .filter(|recorded| *recorded <= capability)
            .map(|recorded| CfgPair {
                name: CFG_ARCH_MIN,
                value: recorded.to_string(),
            }),
    );
    cfgs.push(CfgPair {
        name: CFG_ARCH_TARGET,
        value: arch.sm(),
    });
    if arch.suffix() == Some('a') {
        cfgs.push(CfgPair {
            name: CFG_ARCH_SPECIFIC,
            value: capability.to_string(),
        });
    }
    if matches!(arch.suffix(), Some('a' | 'f')) {
        cfgs.push(CfgPair {
            name: CFG_ARCH_FAMILY_SPECIFIC,
            value: capability.to_string(),
        });
    }
    cfgs
}

/// Render one `cfg(NAME, values("a", "b"))` spec.
fn spec(name: &str, values: impl IntoIterator<Item = String>) -> String {
    let rendered: Vec<String> = values
        .into_iter()
        .map(|value| format!("\"{value}\""))
        .collect();
    format!("cfg({name}, values({}))", rendered.join(", "))
}

/// Declare the legal value universe of every name in [`ALL_CFG_NAMES`].
///
/// Each string is the argument to one `--check-cfg`. Without these, a
/// `#[cfg(cuda_arch_min = "80")]` in user code is an *unexpected* cfg name and
/// warns even when the build did set it; with them, only genuine typos and
/// unconfigured-arch builds warn.
pub fn check_cfg_specs() -> Vec<String> {
    let capabilities = || recorded_capabilities().into_iter().map(|c| c.to_string());
    let mut targets: Vec<String> = RECORDED_PTX_FLOORS
        .iter()
        .map(|entry| match entry.suffix {
            Some(suffix) => format!("sm_{}{suffix}", entry.capability),
            None => format!("sm_{}", entry.capability),
        })
        .collect();
    targets.sort();
    targets.dedup();
    vec![
        spec(CFG_ARCH, capabilities()),
        spec(CFG_ARCH_MIN, capabilities()),
        spec(CFG_ARCH_TARGET, targets),
        spec(
            CFG_ARCH_SPECIFIC,
            recorded_capabilities_with_suffix(&['a'])
                .into_iter()
                .map(|c| c.to_string()),
        ),
        spec(
            CFG_ARCH_FAMILY_SPECIFIC,
            recorded_capabilities_with_suffix(&['a', 'f'])
                .into_iter()
                .map(|c| c.to_string()),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values_for(arch: &str, name: &str) -> Vec<String> {
        arch_cfgs(&arch.parse::<CudaArch>().unwrap())
            .into_iter()
            .filter(|pair| pair.name == name)
            .map(|pair| pair.value)
            .collect()
    }

    #[test]
    fn min_set_is_every_recorded_capability_at_or_below_the_target() {
        assert_eq!(
            values_for("sm_86", CFG_ARCH_MIN),
            ["70", "72", "75", "80", "86"]
        );
        assert_eq!(values_for("sm_86", CFG_ARCH), ["86"]);
        assert!(values_for("sm_86", CFG_ARCH_SPECIFIC).is_empty());
        assert!(values_for("sm_86", CFG_ARCH_FAMILY_SPECIFIC).is_empty());
    }

    /// Numeric ordering, as in CUDA: consumer `sm_120` claims `cuda_arch_min =
    /// "100"` even though it is not a datacenter Blackwell part. Features that
    /// really need sm_100 must gate on the target or family names instead.
    #[test]
    fn ordering_is_numeric_so_sm_120_satisfies_min_100() {
        let min = values_for("sm_120", CFG_ARCH_MIN);
        assert!(min.contains(&"100".to_string()));
        assert!(min.contains(&"110".to_string()));
        assert_eq!(min.last().unwrap(), "120");
        assert!(values_for("sm_120", CFG_ARCH_SPECIFIC).is_empty());
        assert_eq!(values_for("sm_120", CFG_ARCH_TARGET), ["sm_120"]);
    }

    #[test]
    fn arch_specific_suffix_sets_both_specific_names() {
        assert_eq!(values_for("sm_90a", CFG_ARCH_SPECIFIC), ["90"]);
        assert_eq!(values_for("sm_90a", CFG_ARCH_FAMILY_SPECIFIC), ["90"]);
        assert_eq!(values_for("sm_90a", CFG_ARCH), ["90"]);
    }

    #[test]
    fn family_suffix_sets_only_the_family_name() {
        assert!(values_for("sm_100f", CFG_ARCH_SPECIFIC).is_empty());
        assert_eq!(values_for("sm_100f", CFG_ARCH_FAMILY_SPECIFIC), ["100"]);
    }

    #[test]
    fn target_value_is_normalized_to_the_sm_spelling() {
        assert_eq!(values_for("compute_90a", CFG_ARCH_TARGET), ["sm_90a"]);
        assert_eq!(values_for("compute_120", CFG_ARCH_TARGET), ["sm_120"]);
    }

    #[test]
    fn rendered_pairs_quote_their_values() {
        let pair = CfgPair {
            name: CFG_ARCH_MIN,
            value: "80".to_string(),
        };
        assert_eq!(pair.render(), "cuda_arch_min=\"80\"");
    }

    /// The two halves of the contract must agree: anything `arch_cfgs` emits
    /// has to be inside the universe `check_cfg_specs` declares, or rustc
    /// warns about a cfg the wrapper itself set.
    #[test]
    fn every_emitted_value_is_inside_its_declared_universe() {
        let specs = check_cfg_specs();
        assert_eq!(specs.len(), ALL_CFG_NAMES.len());
        let universe = |name: &str| -> String {
            specs
                .iter()
                .find(|spec| spec.starts_with(&format!("cfg({name},")))
                .unwrap_or_else(|| panic!("no --check-cfg spec for {name}"))
                .clone()
        };
        for entry in RECORDED_PTX_FLOORS {
            let arch = CudaArch::new(entry.capability, entry.suffix).unwrap();
            for pair in arch_cfgs(&arch) {
                let spec = universe(pair.name);
                assert!(
                    spec.contains(&format!("\"{}\"", pair.value)),
                    "{}: {} not declared in {spec}",
                    arch,
                    pair.render(),
                );
            }
        }
    }

    #[test]
    fn check_cfg_specs_cover_exactly_the_owned_names() {
        for (name, spec) in ALL_CFG_NAMES.iter().zip(check_cfg_specs()) {
            assert!(spec.starts_with(&format!("cfg({name}, values(")), "{spec}");
            assert!(spec.ends_with("))"), "{spec}");
        }
    }
}
