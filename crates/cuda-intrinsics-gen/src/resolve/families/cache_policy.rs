/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CachePolicyForm, CachePolicyPriority, CachePolicySecondaryPriority,
    ImportedIntrinsic, IntrinsicBackend, IntrinsicSource, OverlayIntrinsic,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};

use crate::resolve::guards::*;

pub(in crate::resolve) struct CachePolicyRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    dialect_op_type: &'static str,
    modifiers: &'static [&'static str],
}

pub(in crate::resolve) const CACHE_POLICY_ISA_SECTION: &str =
    "9.7.9.19 Data Movement and Conversion Instructions: createpolicy";
pub(in crate::resolve) const CACHE_POLICY_ISA_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy";

/// Returns the closed recipe for one `createpolicy` variant.
///
/// Every (form, primary, secondary) combination is its own intrinsic because
/// the qualifiers are compile-time instruction modifiers.
pub(in crate::resolve) fn cache_policy_recipe(
    form: CachePolicyForm,
    primary: CachePolicyPriority,
    secondary: CachePolicySecondaryPriority,
) -> CachePolicyRecipe {
    use CachePolicyPriority as Primary;
    use CachePolicySecondaryPriority as Secondary;
    let CachePolicyForm::Fractional = form;
    match (primary, secondary) {
        (Primary::EvictLast, Secondary::EvictUnchanged) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_last",
            abi_id: "i1039",
            operation_key: "memory.cache_policy.l2.fractional.evict_last",
            dialect_op_type: "CreatepolicyFractionalEvictLastOp",
            modifiers: &["fractional", "L2::evict_last", "b64"],
        },
        (Primary::EvictNormal, Secondary::EvictUnchanged) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_normal",
            abi_id: "i1040",
            operation_key: "memory.cache_policy.l2.fractional.evict_normal",
            dialect_op_type: "CreatepolicyFractionalEvictNormalOp",
            modifiers: &["fractional", "L2::evict_normal", "b64"],
        },
        (Primary::EvictFirst, Secondary::EvictUnchanged) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_first",
            abi_id: "i1041",
            operation_key: "memory.cache_policy.l2.fractional.evict_first",
            dialect_op_type: "CreatepolicyFractionalEvictFirstOp",
            modifiers: &["fractional", "L2::evict_first", "b64"],
        },
        (Primary::EvictUnchanged, Secondary::EvictUnchanged) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_unchanged",
            abi_id: "i1042",
            operation_key: "memory.cache_policy.l2.fractional.evict_unchanged",
            dialect_op_type: "CreatepolicyFractionalEvictUnchangedOp",
            modifiers: &["fractional", "L2::evict_unchanged", "b64"],
        },
        (Primary::EvictLast, Secondary::EvictFirst) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_last_evict_first",
            abi_id: "i1043",
            operation_key: "memory.cache_policy.l2.fractional.evict_last.evict_first",
            dialect_op_type: "CreatepolicyFractionalEvictLastEvictFirstOp",
            modifiers: &["fractional", "L2::evict_last", "L2::evict_first", "b64"],
        },
        (Primary::EvictNormal, Secondary::EvictFirst) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_normal_evict_first",
            abi_id: "i1044",
            operation_key: "memory.cache_policy.l2.fractional.evict_normal.evict_first",
            dialect_op_type: "CreatepolicyFractionalEvictNormalEvictFirstOp",
            modifiers: &["fractional", "L2::evict_normal", "L2::evict_first", "b64"],
        },
        (Primary::EvictFirst, Secondary::EvictFirst) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_first_evict_first",
            abi_id: "i1045",
            operation_key: "memory.cache_policy.l2.fractional.evict_first.evict_first",
            dialect_op_type: "CreatepolicyFractionalEvictFirstEvictFirstOp",
            modifiers: &["fractional", "L2::evict_first", "L2::evict_first", "b64"],
        },
        (Primary::EvictUnchanged, Secondary::EvictFirst) => CachePolicyRecipe {
            id: "createpolicy_fractional_evict_unchanged_evict_first",
            abi_id: "i1046",
            operation_key: "memory.cache_policy.l2.fractional.evict_unchanged.evict_first",
            dialect_op_type: "CreatepolicyFractionalEvictUnchangedEvictFirstOp",
            modifiers: &[
                "fractional",
                "L2::evict_unchanged",
                "L2::evict_first",
                "b64",
            ],
        },
    }
}

pub(in crate::resolve) fn validate_cache_policy_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let contract = policy
        .cache_policy
        .as_ref()
        .with_context(|| format!("{} has no closed cache-policy contract", policy.id))?;
    let recipe = cache_policy_recipe(contract.form, contract.primary, contract.secondary);
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} cache-policy identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "cache_policy"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == ["f32"]
            && policy.rust_result == "u64"
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::cache_policy::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::cache_policy::{}", recipe.id)],
        "{} must preserve its reviewed safe cache-policy API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == format!("nvvm.{}", recipe.id)
            && policy.dialect_operands == ["f32"]
            && policy.dialect_results == ["i64"]
            && policy.lowering == "generated_cache_policy_inline_ptx",
        "{} is outside the closed cache-policy dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.4"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "u64"
            && policy.targets == "all",
        "{} cache-policy effects, carrier, or target floor disagree",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == CACHE_POLICY_ISA_SECTION
            && policy.ptx_isa_url == CACHE_POLICY_ISA_URL,
        "{} cache-policy PTX provenance does not match its reviewed instruction section",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "createpolicy"
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == vec![OperandPattern::Register; 2],
        "{} expected PTX does not match its exact createpolicy instruction",
        policy.id
    );
    ensure!(
        source
            == &IntrinsicSource::PtxNative {
                instruction: contract.ptx_instruction(),
            }
            && declaration.is_none(),
        "{} cache-policy source does not match its PTX-native recipe",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings[0].backend == IntrinsicBackend::LlvmNvptx
            && policy.backend_lowerings[1].backend == IntrinsicBackend::LibNvvm
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.targets.is_none()
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} cache-policy backend route changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cache_policy")?;
    Ok(())
}
