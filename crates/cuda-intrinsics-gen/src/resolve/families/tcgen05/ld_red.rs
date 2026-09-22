/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `tcgen05.ld.red`: tensor-memory loads that also return a min/max
//! reduction across the loaded registers.

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayIntrinsic,
    RuntimeValidation, Tcgen05, Tcgen05Adapter, Tcgen05Admission, Tcgen05LdMultiplicity,
    Tcgen05LdRed, Tcgen05LdRedAdmissionVariant, Tcgen05LdRedElement, Tcgen05LdRedOp,
    Tcgen05LdShape, Tcgen05Operation, Tcgen05SourceContract,
};
use crate::ptx::OperandPattern;
use anyhow::{Result, ensure};

use super::*;
use crate::resolve::guards::*;

/// Both backends admit only the architecture-specific targets that CUDA 13.3
/// ptxas accepts for `tcgen05.ld.red`; plain and `sm_100a` targets reject it.
pub(in crate::resolve) const TCGEN05_LD_RED_TARGETS: &str = "sm_103a|sm_110a";
pub(in crate::resolve) const TCGEN05_LD_RED_MINIMUM_PTX: &str = "8.8";

pub(in crate::resolve) const TCGEN05_LD_RED_SHAPES: [Tcgen05LdShape; 2] =
    [Tcgen05LdShape::M32x32b, Tcgen05LdShape::M16x32bx2];

/// `tcgen05.ld.red` has no `.x1` form.
pub(in crate::resolve) const TCGEN05_LD_RED_MULTIPLICITIES: [Tcgen05LdMultiplicity; 7] = [
    Tcgen05LdMultiplicity::X2,
    Tcgen05LdMultiplicity::X4,
    Tcgen05LdMultiplicity::X8,
    Tcgen05LdMultiplicity::X16,
    Tcgen05LdMultiplicity::X32,
    Tcgen05LdMultiplicity::X64,
    Tcgen05LdMultiplicity::X128,
];

/// Canonical admission order: shape, repetition, element, reduction, then
/// the f32-only `.abs` and `.NaN` modifiers.
pub(in crate::resolve) fn tcgen05_ld_red_variants() -> Vec<Tcgen05LdRed> {
    let mut variants = Vec::new();
    for shape in TCGEN05_LD_RED_SHAPES {
        for multiplicity in TCGEN05_LD_RED_MULTIPLICITIES {
            for element in [
                Tcgen05LdRedElement::U32,
                Tcgen05LdRedElement::S32,
                Tcgen05LdRedElement::F32,
            ] {
                for op in [Tcgen05LdRedOp::Min, Tcgen05LdRedOp::Max] {
                    let flags: &[(bool, bool)] = if element == Tcgen05LdRedElement::F32 {
                        &[(false, false), (false, true), (true, false), (true, true)]
                    } else {
                        &[(false, false)]
                    };
                    for &(abs, nan) in flags {
                        variants.push(Tcgen05LdRed {
                            shape,
                            multiplicity,
                            op,
                            element,
                            abs,
                            nan,
                        });
                    }
                }
            }
        }
    }
    variants
}

pub(in crate::resolve) fn tcgen05_ld_red_register_count(ld_red: Tcgen05LdRed) -> usize {
    ld_red.shape.register_multiplier() * ld_red.multiplicity.count()
}

fn tcgen05_ld_red_flag_suffix(ld_red: Tcgen05LdRed, separator: char) -> String {
    let mut suffix = String::new();
    if ld_red.abs {
        suffix.push(separator);
        suffix.push_str("abs");
    }
    if ld_red.nan {
        suffix.push(separator);
        suffix.push_str("nan");
    }
    suffix
}

pub(in crate::resolve) fn tcgen05_ld_red_id(ld_red: Tcgen05LdRed) -> String {
    format!(
        "tcgen05_ld_red_{}_{}_{}{}_{}",
        tcgen05_ld_shape_name(ld_red.shape),
        tcgen05_ld_multiplicity_name(ld_red.multiplicity),
        ld_red.op.name(),
        tcgen05_ld_red_flag_suffix(ld_red, '_'),
        ld_red.element.ptx_name()
    )
}

pub(in crate::resolve) fn tcgen05_ld_red_operation_key(ld_red: Tcgen05LdRed) -> String {
    format!(
        "tcgen05.ld.red.{}.{}.{}{}.{}",
        tcgen05_ld_shape_name(ld_red.shape),
        tcgen05_ld_multiplicity_name(ld_red.multiplicity),
        ld_red.op.name(),
        tcgen05_ld_red_flag_suffix(ld_red, '.'),
        ld_red.element.ptx_name()
    )
}

pub(in crate::resolve) fn tcgen05_ld_red_source_record(ld_red: Tcgen05LdRed) -> String {
    format!(
        "int_nvvm_tcgen05_ld_red_{}_{}_{}",
        tcgen05_ld_shape_name(ld_red.shape),
        tcgen05_ld_multiplicity_name(ld_red.multiplicity),
        ld_red.element.llvm_suffix()
    )
}

pub(in crate::resolve) fn tcgen05_ld_red_llvm_symbol(ld_red: Tcgen05LdRed) -> String {
    format!(
        "llvm.nvvm.tcgen05.ld.red.{}.{}.{}",
        tcgen05_ld_shape_name(ld_red.shape),
        tcgen05_ld_multiplicity_name(ld_red.multiplicity),
        ld_red.element.llvm_suffix()
    )
}

pub(in crate::resolve) fn tcgen05_ld_red_rust_result(ld_red: Tcgen05LdRed) -> String {
    let element = ld_red.element.rust_type();
    format!(
        "([{element}; {}], {element})",
        tcgen05_ld_red_register_count(ld_red)
    )
}

/// The pinned TableGen dump shares the overloaded i32 ld/st vector records
/// and names one anonymous vector record per f32 register count, plus one
/// scalar record per reduction type.
pub(in crate::resolve) fn tcgen05_ld_red_llvm_results(ld_red: Tcgen05LdRed) -> Vec<String> {
    let count = tcgen05_ld_red_register_count(ld_red);
    match ld_red.element {
        Tcgen05LdRedElement::U32 | Tcgen05LdRedElement::S32 => vec![
            tcgen05_overloaded_data_token(count),
            "anonymous_10027".into(),
        ],
        Tcgen05LdRedElement::F32 => {
            let vector = match count {
                2 => 10022,
                4 => 10031,
                8 => 10038,
                16 => 10045,
                32 => 10052,
                64 => 10059,
                128 => 10066,
                other => unreachable!("tcgen05.ld.red f32 count {other} has no imported record"),
            };
            vec![format!("anonymous_{vector}"), "anonymous_10023".into()]
        }
    }
}

fn tcgen05_ld_red_has_half_split_offset(ld_red: Tcgen05LdRed) -> bool {
    ld_red.shape == Tcgen05LdShape::M16x32bx2
}

pub(in crate::resolve) fn tcgen05_ld_red_llvm_arguments(ld_red: Tcgen05LdRed) -> Vec<String> {
    let mut arguments = vec!["tmem_ptr".to_owned()];
    if tcgen05_ld_red_has_half_split_offset(ld_red) {
        arguments.push("i64".into());
    }
    arguments.push("i32".into());
    if ld_red.element == Tcgen05LdRedElement::F32 {
        arguments.extend(["i1".into(), "i1".into()]);
    }
    arguments
}

fn tcgen05_ld_red_imported_properties(ld_red: Tcgen05LdRed) -> Vec<String> {
    let immediate_count = tcgen05_ld_red_llvm_arguments(ld_red).len() - 1;
    (1..=immediate_count)
        .map(|index| format!("ImmArg<arg{index}>"))
        .chain(
            ["IntrArgMemOnly", "IntrConvergent", "NoCapture<arg0>"]
                .into_iter()
                .map(Into::into),
        )
        .collect()
}

pub(in crate::resolve) fn tcgen05_ld_red_op_type(ld_red: Tcgen05LdRed) -> String {
    let multiplicity = tcgen05_ld_multiplicity_name(ld_red.multiplicity)
        .strip_prefix('x')
        .unwrap();
    let op = match ld_red.op {
        Tcgen05LdRedOp::Min => "Min",
        Tcgen05LdRedOp::Max => "Max",
    };
    let element = match ld_red.element {
        Tcgen05LdRedElement::U32 => "U32",
        Tcgen05LdRedElement::S32 => "S32",
        Tcgen05LdRedElement::F32 => "F32",
    };
    format!(
        "Tcgen05LdRed{}X{multiplicity}{op}{}{}{element}Op",
        tcgen05_ld_shape_name(ld_red.shape),
        if ld_red.abs { "Abs" } else { "" },
        if ld_red.nan { "Nan" } else { "" },
    )
}

pub(in crate::resolve) fn tcgen05_ld_red_modifiers(ld_red: Tcgen05LdRed) -> Vec<String> {
    let mut modifiers: Vec<String> = vec![
        "ld".into(),
        "red".into(),
        "sync".into(),
        "aligned".into(),
        tcgen05_ld_shape_name(ld_red.shape).into(),
        tcgen05_ld_multiplicity_name(ld_red.multiplicity).into(),
        ld_red.op.name().into(),
    ];
    if ld_red.abs {
        modifiers.push("abs".into());
    }
    if ld_red.nan {
        modifiers.push("NaN".into());
    }
    modifiers.push(ld_red.element.ptx_name().into());
    modifiers
}

pub(in crate::resolve) fn tcgen05_ld_red_operands(ld_red: Tcgen05LdRed) -> Vec<OperandPattern> {
    let mut operands = vec![
        OperandPattern::RegisterList {
            length: tcgen05_ld_red_register_count(ld_red),
        },
        OperandPattern::Register,
        OperandPattern::Address,
    ];
    if tcgen05_ld_red_has_half_split_offset(ld_red) {
        operands.push(OperandPattern::Immediate);
    }
    operands
}

fn tcgen05_ld_red_summary(ld_red: Tcgen05LdRed) -> String {
    let count = tcgen05_ld_red_register_count(ld_red);
    let extreme = match ld_red.op {
        Tcgen05LdRedOp::Min => "minimum",
        Tcgen05LdRedOp::Max => "maximum",
    };
    let (element, reduced) = match ld_red.element {
        Tcgen05LdRedElement::U32 => ("unsigned 32-bit", "value"),
        Tcgen05LdRedElement::S32 => ("signed 32-bit", "value"),
        Tcgen05LdRedElement::F32 => (
            "f32",
            if ld_red.abs {
                "absolute value"
            } else {
                "value"
            },
        ),
    };
    let nan = if ld_red.nan {
        ", propagating NaN inputs"
    } else {
        ""
    };
    format!(
        "Loads {count} {element} values from tensor memory and reduces them to their {extreme} {reduced}{nan}."
    )
}

pub(in crate::resolve) fn materialize_tcgen05_ld_red_variant(
    base: &OverlayIntrinsic,
    admission: &Tcgen05Admission,
    variant: &Tcgen05LdRedAdmissionVariant,
) -> OverlayIntrinsic {
    let ld_red = Tcgen05LdRed {
        shape: variant.shape,
        multiplicity: variant.multiplicity,
        op: variant.op,
        element: variant.element,
        abs: variant.abs,
        nan: variant.nan,
    };
    let id = tcgen05_ld_red_id(ld_red);
    let register_count = tcgen05_ld_red_register_count(ld_red);
    let rust_result = tcgen05_ld_red_rust_result(ld_red);
    let has_half_split_offset = tcgen05_ld_red_has_half_split_offset(ld_red);
    let mut record = base.clone();
    record.id = id.clone();
    record.abi_id = variant.abi_id.clone();
    record.operation_key = tcgen05_ld_red_operation_key(ld_red);
    record.source_record = Some(tcgen05_ld_red_source_record(ld_red));
    record.rust_name = id.clone();
    record.rust_arguments = if has_half_split_offset {
        vec!["u32".into(), "i64".into()]
    } else {
        vec!["u32".into()]
    };
    record.rust_result = rust_result.clone();
    record.must_use = true;
    record.public_rust_path = format!("cuda_intrinsics::tcgen05::{id}");
    record.compatibility_rust_paths = vec![format!(
        "cuda_device::tcgen05::{}",
        if has_half_split_offset {
            format!("__{id}")
        } else {
            id.clone()
        }
    )];
    record.dialect_op_type = tcgen05_ld_red_op_type(ld_red);
    record.dialect_op_name = format!("nvvm.{id}");
    record.dialect_operands = if has_half_split_offset {
        vec!["i32".into(), "i64".into()]
    } else {
        vec!["i32".into()]
    };
    record.dialect_results = vec![ld_red.element.dialect_type().into(); register_count + 1];
    record.llvm_symbol = Some(tcgen05_ld_red_llvm_symbol(ld_red));
    record.resolved_llvm_symbol = None;
    record.llvm_arguments = tcgen05_ld_red_llvm_arguments(ld_red);
    record.llvm_results = tcgen05_ld_red_llvm_results(ld_red);
    record.minimum_ptx = TCGEN05_LD_RED_MINIMUM_PTX.into();
    record.targets = TCGEN05_LD_RED_TARGETS.into();
    record.ptx_isa_version = TCGEN05_LD_RED_MINIMUM_PTX.into();
    record.ptx_isa_section = "Tensor Memory tcgen05 instructions: tcgen05.ld".into();
    record.ptx_isa_url =
        "https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld"
            .into();
    record.ptx_result = rust_result;
    record.execution_scope = Tcgen05Operation::LdRed.execution_scope().into();
    for (route, profile) in record.backend_lowerings.iter_mut().zip([
        &admission.ld_red_llvm_evidence_profile,
        &admission.ld_red_libnvvm_evidence_profile,
    ]) {
        route.evidence_profile = profile
            .as_ref()
            .expect("validated tcgen05 reducing-load evidence profile")
            .clone();
        route.minimum_ptx = Some(TCGEN05_LD_RED_MINIMUM_PTX.into());
        route.targets =
            (route.backend == IntrinsicBackend::LibNvvm).then(|| TCGEN05_LD_RED_TARGETS.into());
    }
    record.tcgen05 = Some(Tcgen05 {
        operation: Tcgen05Operation::LdRed,
        cp: None,
        ld: None,
        ld_red: Some(ld_red),
        st: None,
        mma: None,
        adapter: if has_half_split_offset {
            Tcgen05Adapter::TmemHalfSplitOffsetInjectReductionToRegistersAndValue
        } else {
            Tcgen05Adapter::TmemInjectReductionToRegistersAndValue
        },
        source_contract: Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        runtime_validation: admission.runtime_validation,
    });
    record.expected_ptx.modifiers = tcgen05_ld_red_modifiers(ld_red);
    record.expected_ptx.operands = tcgen05_ld_red_operands(ld_red);
    record.summary = tcgen05_ld_red_summary(ld_red);
    record
}

pub(in crate::resolve) fn validate_tcgen05_ld_red_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tcgen05: &Tcgen05,
    ld_red: Tcgen05LdRed,
) -> Result<()> {
    ensure!(
        tcgen05_ld_red_variants().contains(&ld_red),
        "{} has an unsupported tcgen05 reducing-load identity",
        policy.id
    );
    let id = tcgen05_ld_red_id(ld_red);
    let source_record = tcgen05_ld_red_source_record(ld_red);
    let llvm_symbol = tcgen05_ld_red_llvm_symbol(ld_red);
    let rust_result = tcgen05_ld_red_rust_result(ld_red);
    let register_count = tcgen05_ld_red_register_count(ld_red);
    let has_half_split_offset = tcgen05_ld_red_has_half_split_offset(ld_red);
    let llvm_arguments = tcgen05_ld_red_llvm_arguments(ld_red);
    let llvm_results = tcgen05_ld_red_llvm_results(ld_red);
    let expected_rust_arguments = if has_half_split_offset {
        vec!["u32", "i64"]
    } else {
        vec!["u32"]
    };
    let expected_dialect_operands = if has_half_split_offset {
        vec!["i32", "i64"]
    } else {
        vec!["i32"]
    };
    let compatibility_name = if has_half_split_offset {
        format!("__{id}")
    } else {
        id.clone()
    };
    ensure!(
        policy.id == id
            && policy.operation_key == tcgen05_ld_red_operation_key(ld_red)
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == source_record
            && declaration.llvm_name == llvm_symbol,
        "{} tcgen05 reducing-load identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == id
            && policy.rust_arguments == expected_rust_arguments
            && policy.rust_result == rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{id}")
            && policy.compatibility_rust_paths
                == [format!("cuda_device::tcgen05::{compatibility_name}")],
        "{} tcgen05 reducing-load Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == tcgen05_ld_red_op_type(ld_red)
            && policy.dialect_op_name == format!("nvvm.{id}")
            && policy.dialect_operands == expected_dialect_operands
            && policy.dialect_results == vec![ld_red.element.dialect_type(); register_count + 1]
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results == llvm_results
            && declaration.arguments == llvm_arguments
            && declaration.results == llvm_results
            && declaration.classes
                == [
                    "SDPatternOperator",
                    "Intrinsic",
                    "DefaultAttrsIntrinsic",
                    "DefaultAttrsIntrinsicFlags",
                    "NVVM_TCGEN05_LD_RED",
                ]
            && declaration.properties == tcgen05_ld_red_imported_properties(ld_red)
            && declaration.selections.is_empty()
            && policy.lowering == "generated_tcgen05",
        "{} tcgen05 reducing-load carrier or imported declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read"
            && policy.convergent
            && policy.execution_scope == Tcgen05Operation::LdRed.execution_scope()
            && tcgen05.operation == Tcgen05Operation::LdRed
            && tcgen05.cp.is_none()
            && tcgen05.ld.is_none()
            && tcgen05.ld_red == Some(ld_red)
            && tcgen05.st.is_none()
            && tcgen05.mma.is_none()
            && tcgen05.adapter
                == if has_half_split_offset {
                    Tcgen05Adapter::TmemHalfSplitOffsetInjectReductionToRegistersAndValue
                } else {
                    Tcgen05Adapter::TmemInjectReductionToRegistersAndValue
                }
            && tcgen05.source_contract == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 reducing-load semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == TCGEN05_LD_RED_MINIMUM_PTX
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LD_RED_TARGETS
            && policy.ptx_isa_version == TCGEN05_LD_RED_MINIMUM_PTX
            && policy.ptx_result == rust_result
            && policy.expected_ptx.mnemonic == "tcgen05"
            && policy.expected_ptx.modifiers == tcgen05_ld_red_modifiers(ld_red)
            && policy.expected_ptx.operands == tcgen05_ld_red_operands(ld_red),
        "{} tcgen05 reducing-load target or PTX contract changed",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings[0].backend == IntrinsicBackend::LlvmNvptx
            && policy.backend_lowerings[1].backend == IntrinsicBackend::LibNvvm
            && policy.backend_lowerings[0].targets.is_none()
            && policy.backend_lowerings[1].targets.as_deref() == Some(TCGEN05_LD_RED_TARGETS)
            && policy.backend_lowerings.iter().all(|route| {
                route.mechanism == BackendLoweringMechanism::InlinePtx
                    && route.minimum_ptx.as_deref() == Some(TCGEN05_LD_RED_MINIMUM_PTX)
                    && route.minimum_sm.is_none()
                    && !route.evidence_profile.trim().is_empty()
            }),
        "{} tcgen05 reducing-load backend route changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "tcgen05 reducing load")?;
    Ok(())
}
