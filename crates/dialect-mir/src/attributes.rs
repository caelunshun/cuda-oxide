/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes belonging to the MIR dialect.

use std::hash::{Hash, Hasher};

use pliron::attribute::Attribute;
use pliron::builtin::attr_interfaces::{FloatAttr, TypedAttrInterface};
use pliron::context::Context;
use pliron::derive::{attr_interface_impl, pliron_attr};
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apfloat::{self, Float, GetSemantics};

use crate::types::MirFP16Type;

/// MIR cast kind — preserves the semantic intent of the cast from Rust MIR.
///
/// The lowering dispatches on this to pick the correct LLVM instruction,
/// rather than guessing from source/destination types.
#[pliron_attr(name = "mir.cast_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum MirCastKindAttr {
    IntToInt,
    IntToFloat,
    FloatToInt,
    FloatToFloat,
    PtrToPtr,
    FnPtrToPtr,
    PointerExposeAddress,
    PointerWithExposedProvenance,
    Transmute,
    PointerCoercionUnsize,
    PointerCoercionMutToConst,
    PointerCoercionArrayToPointer,
    PointerCoercionReifyFnPointer,
    PointerCoercionUnsafeFnPointer,
    PointerCoercionClosureFnPointer,
    Subtype,
}

/// The Rust semantic boundary that authorizes a pointer-kind transition.
///
/// Ordinary MIR operations may preserve a pointer kind or deliberately erase
/// it, but they may not recover or change a concrete Rust pointer category.
/// A `mir.cast` that does establish a new concrete category must carry one of
/// these authorities so the provenance-changing boundary remains explicit and
/// auditable in the dialect.
#[pliron_attr(name = "mir.pointer_kind_authority", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum MirPointerKindAuthorityAttr {
    /// A rustc `Rvalue::Ref` (`&place` / `&mut place`).
    Reborrow,
    /// A rustc `Rvalue::AddressOf` (`&raw const` / `&raw mut`).
    RawAddress,
    /// A pointer-producing cast or coercion explicitly present in rustc MIR.
    RustCast,
    /// Materialization of a pointer-valued Rust constant, static, or promoted
    /// allocation at the exact type declared by rustc.
    StaticAddress,
    /// Adaptation to an exact Rust function or intrinsic ABI type.
    AbiBoundary,
    /// A user-authored inline-assembly output assigned the exact destination
    /// type supplied by rustc MIR. This records the unsafe source boundary;
    /// it does not independently prove that the assembly produced valid bits.
    InlineAsm,
}

/// Boolean attribute for reference mutability.
///
/// Replaces the overloaded `IntegerAttr` pattern with a self-documenting
/// domain-specific attribute.
#[pliron_attr(name = "mir.mutability", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct MutabilityAttr(pub bool);

/// Proven validity facts for one Rust-reference kernel parameter.
///
/// Presence of this attribute means the source argument is a Rust reference
/// proven non-null by `rustc_public`. The payload is the pointee alignment in
/// bytes; it must be a non-zero power of two. LLVM `align 1` is redundant and
/// is omitted during export, but retaining `1` here keeps the proof complete.
///
/// The attribute is stored on `mir.func` under a source-argument-indexed key.
/// It intentionally carries no aliasing or dereferenceability promise.
#[pliron_attr(
    name = "mir.reference_param_validity",
    format = "$0",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct ReferenceParamValidityAttr(pub u64);

/// Structural field index for aggregate access ops
/// (`mir.extract_field`, `mir.insert_field`, `mir.field_addr`, `mir.enum_payload`).
#[pliron_attr(name = "mir.field_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct FieldIndexAttr(pub u32);

/// Enum variant index for variant-level ops
/// (`mir.construct_enum`, `mir.enum_payload`).
#[pliron_attr(name = "mir.variant_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct VariantIndexAttr(pub u32);

/// The unroll factor carried by a [`MirUnrollHintOp`](crate::ops::MirUnrollHintOp).
///
/// `#[unroll]` / `#[unroll(N)]` written on a loop makes the frontend plant a
/// `mir.unroll_hint` op inside that loop's body; this attribute is the factor it
/// carries, and the loop-unroll pass reads it to decide how to unroll that one
/// loop:
///
/// * `0` -- **full unroll**: if the loop's trip count is a compile-time
///   constant, unroll it completely, so the induction variable becomes a literal
///   in each copy (this is what lets index arithmetic such as `i & 3` fold to a
///   constant).
/// * `n >= 2` -- **unroll by `n`**: do `n` copies of the body per trip, leaving
///   a remainder loop when `n` does not divide the trip count.
#[pliron_attr(name = "mir.unroll", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct UnrollAttr(pub u32);

/// An LLVM loop-unroll request recorded on a loop's latch terminator
/// ([`MirGotoOp`](crate::ops::MirGotoOp) or
/// [`MirCondBranchOp`](crate::ops::MirCondBranchOp)).
///
/// `#[llvm_unroll]` / `#[llvm_unroll(N)]` makes the frontend plant a
/// `mir.llvm_unroll_hint` op inside the loop body; `mir-transforms` maps that
/// hint back to its loop and moves the request onto the latch, where lowering
/// can turn it into `!llvm.loop` metadata. cuda-oxide does no unrolling of its
/// own for this attribute -- LLVM's loop-unroll pass reads the metadata.
///
/// * `factor == 0` -- `llvm.loop.unroll.full`.
/// * `factor >= 2` -- `llvm.loop.unroll.count`, which applies to runtime trip
///   counts too.
///
/// `group` identifies the loop the latch belongs to. A loop with several
/// latches must reference one shared `!llvm.loop` node, so every latch of one
/// loop carries the same group.
#[pliron_attr(
    name = "mir.llvm_loop_unroll",
    format = "`<` $factor `, ` $group `>`",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct LlvmLoopUnrollAttr {
    pub factor: u32,
    pub group: u32,
}

/// Operation attribute key carrying [`LlvmLoopUnrollAttr`] on a latch
/// terminator.
///
/// Both `mir.goto` and `mir.cond_br` can be a latch, and pliron's
/// attribute-name registry is global -- one name may be declared by one op only
/// -- so the request is keyed rather than declared per op. Use
/// [`set_llvm_loop_unroll`](crate::ops::control_flow::set_llvm_loop_unroll) and
/// [`llvm_loop_unroll`](crate::ops::control_flow::llvm_loop_unroll) instead of
/// the key directly.
pub const LLVM_LOOP_UNROLL_ATTR_KEY: &str = "llvm_loop_unroll";

/// Marks an aggregate that exists only to adapt one compiler-owned
/// multi-result operation to a Rust aggregate return ABI.
///
/// The marker is intentionally attached by the MIR importer, not inferred by
/// an optimisation pass. That lets the forwarding pass distinguish this
/// compiler-created boundary from an ordinary user aggregate and fail closed
/// whenever the exact producer/store/projection shape is not preserved.
#[pliron_attr(name = "mir.compiler_result_bundle", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CompilerResultBundleAttr(pub bool);

/// Operation attribute key carrying [`CompilerResultBundleAttr`].
pub const COMPILER_RESULT_BUNDLE_ATTR_KEY: &str = "compiler_result_bundle";

/// IEEE 754 binary16 floating-point attribute for Rust MIR `f16` constants.
#[pliron_attr(name = "mir.fp16_attr", format = "$0", verifier = "succ")]
#[derive(PartialEq, Clone, Debug)]
pub struct MirFP16Attr(pub apfloat::Half);

impl MirFP16Attr {
    pub fn from_bits(bits: u16) -> Self {
        MirFP16Attr(<apfloat::Half as Float>::from_bits(bits as u128))
    }

    pub fn to_bits(&self) -> u16 {
        self.0.to_bits() as u16
    }
}

impl Hash for MirFP16Attr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }
}

impl Typed for MirFP16Attr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        MirFP16Type::get(ctx).into()
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for MirFP16Attr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        MirFP16Type::get(ctx).into()
    }
}

#[attr_interface_impl]
impl FloatAttr for MirFP16Attr {
    fn get_inner(&self) -> &dyn apfloat::DynFloat {
        &self.0
    }

    fn build_from(&self, df: Box<dyn apfloat::DynFloat>) -> Box<dyn FloatAttr> {
        let df = df
            .downcast::<apfloat::Half>()
            .expect("Expected a half precision float");
        Box::new(MirFP16Attr(*df))
    }

    fn get_semantics(&self) -> apfloat::Semantics {
        Self::get_semantics_static()
    }

    fn get_semantics_static() -> apfloat::Semantics
    where
        Self: Sized,
    {
        <apfloat::Half as GetSemantics>::get_semantics()
    }
}

pub fn register(ctx: &mut Context) {
    MirCastKindAttr::register(ctx);
    MirPointerKindAuthorityAttr::register(ctx);
    MutabilityAttr::register(ctx);
    ReferenceParamValidityAttr::register(ctx);
    FieldIndexAttr::register(ctx);
    VariantIndexAttr::register(ctx);
    UnrollAttr::register(ctx);
    LlvmLoopUnrollAttr::register(ctx);
    CompilerResultBundleAttr::register(ctx);
    MirFP16Attr::register(ctx);
}
