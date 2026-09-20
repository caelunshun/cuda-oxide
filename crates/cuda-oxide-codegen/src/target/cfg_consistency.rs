/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check the architecture `#[cfg]` selected against the one actually built.
//!
//! `cargo oxide` resolves a target, injects `cuda_arch*` cfgs for it, and
//! reports that architecture back on `CUDA_OXIDE_INTERNAL_CFG_ARCH`. rustc
//! then evaluates the `#[cfg]`s and discards the unselected arms long before
//! this backend sees any IR -- so by the time target selection runs, the
//! arch-conditional decisions are already baked in and cannot be revisited.
//!
//! Usually the two agree. One path can separate them:
//!
//! ```text
//! cargo oxide run  -->  detected GPU sm_86  -->  cfgs for sm_86
//!                                                   |
//!               kernel needs tcgen05 -----------> backend rejects the hint,
//!                                                 builds for sm_100a
//! ```
//!
//! The build is still correct -- the module simply will not load on the local
//! GPU, which is the pre-existing behaviour of an advisory hint -- but the
//! `#[cfg(cuda_arch_min = "100")]` arms the user expected were not compiled.
//! That is worth saying out loud, and saying how to fix it (`--arch`).
//!
//! The same mismatch against an *explicit* pin is impossible by construction:
//! the pin sets both the cfgs and the backend's target. If it happens anyway,
//! the plumbing between the two has drifted, so that case is an error rather
//! than a warning.

use cuda_target_spec::CudaArch;

use crate::error::PipelineError;

/// Result of comparing the cfg architecture against the selected target.
#[derive(Debug)]
pub(crate) enum CfgConsistency {
    /// Nothing to report: no cfgs were injected, or they match.
    Consistent,
    /// Advisory: the build is valid but not the one the cfgs describe.
    Warn(String),
    /// The pin that set the cfgs should have set this target too.
    Reject(PipelineError),
}

/// Compare the reported cfg architecture with the selected target.
///
/// `selected_source` is the provenance label target selection returned;
/// `explicit_source` is the label it would have used had the explicit pin
/// won, so equality between them is how this tells "pinned" from "inferred"
/// without threading a second enum through the selectors.
pub(crate) fn check(
    cfg_arch: Option<&str>,
    selected: &CudaArch,
    selected_source: &str,
    explicit_source: &str,
) -> CfgConsistency {
    let Some(cfg_arch) = cfg_arch else {
        return CfgConsistency::Consistent;
    };
    let Ok(parsed) = cfg_arch.parse::<CudaArch>() else {
        // Not fatal: an unreadable report only costs this check, and failing
        // the build over a diagnostic channel would be worse than the bug.
        return CfgConsistency::Warn(format!(
            "warning: could not read the architecture `{cfg_arch}` that arch-conditional code was \
             compiled for, so it was not checked against the built target {}",
            selected.sm()
        ));
    };
    if &parsed == selected {
        return CfgConsistency::Consistent;
    }
    if selected_source == explicit_source {
        return CfgConsistency::Reject(PipelineError::TargetSelection {
            target: selected.sm(),
            reason: format!(
                "arch-conditional code was compiled for {} but the module is being built for {} \
                 (target from {selected_source}); a pinned architecture must produce both, so \
                 this is a cuda-oxide plumbing bug rather than a problem with this crate",
                parsed.sm(),
                selected.sm(),
            ),
        });
    }
    CfgConsistency::Warn(format!(
        "warning: arch-conditional code was compiled for {} (the detected GPU) but this module \
         needs {} and was built for it; pass `--arch {}` to specialize for the built target",
        parsed.sm(),
        selected.sm(),
        selected.sm(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(target: &str) -> CudaArch {
        target.parse().unwrap()
    }

    #[test]
    fn nothing_to_report_without_cfgs_or_on_agreement() {
        assert!(matches!(
            check(
                None,
                &arch("sm_90a"),
                "feature requirement",
                "CUDA_OXIDE_TARGET"
            ),
            CfgConsistency::Consistent
        ));
        assert!(matches!(
            check(
                Some("sm_90a"),
                &arch("sm_90a"),
                "CUDA_OXIDE_TARGET",
                "CUDA_OXIDE_TARGET"
            ),
            CfgConsistency::Consistent
        ));
    }

    #[test]
    fn a_rejected_gpu_hint_warns_and_names_the_fix() {
        let CfgConsistency::Warn(message) = check(
            Some("sm_86"),
            &arch("sm_100a"),
            "feature requirement",
            "CUDA_OXIDE_TARGET",
        ) else {
            panic!("an overridden hint must warn");
        };
        assert!(message.contains("compiled for sm_86"), "{message}");
        assert!(message.contains("needs sm_100a"), "{message}");
        assert!(message.contains("--arch sm_100a"), "{message}");
    }

    /// The pin feeds both halves, so disagreement means the wrapper and the
    /// backend disagree about what was pinned. Fail loudly instead of
    /// emitting code whose `#[cfg]` arms were chosen for another GPU.
    #[test]
    fn disagreeing_with_an_explicit_pin_is_an_error() {
        let CfgConsistency::Reject(error) = check(
            Some("sm_86"),
            &arch("sm_90a"),
            "CUDA_OXIDE_TARGET",
            "CUDA_OXIDE_TARGET",
        ) else {
            panic!("a pinned mismatch must not be downgraded to a warning");
        };
        let PipelineError::TargetSelection { target, reason } = error else {
            panic!("expected a target-selection error");
        };
        assert_eq!(target, "sm_90a");
        assert!(reason.contains("sm_86"), "{reason}");
        assert!(reason.contains("plumbing bug"), "{reason}");
    }

    #[test]
    fn an_unreadable_report_warns_rather_than_failing_the_build() {
        let CfgConsistency::Warn(message) = check(
            Some("not-an-arch"),
            &arch("sm_90a"),
            "feature requirement",
            "CUDA_OXIDE_TARGET",
        ) else {
            panic!("an unparsable report must not fail the build");
        };
        assert!(message.contains("not-an-arch"), "{message}");
        assert!(message.contains("sm_90a"), "{message}");
    }
}
