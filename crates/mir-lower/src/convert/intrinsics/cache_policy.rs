// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering helper for generated `createpolicy` L2 cache-policy operations.
//!
//! Every variant is a pure one-operand instruction: an `f32` fraction in, an
//! opaque 64-bit policy out. The qualifiers are baked into the instruction
//! text, so the helper only needs the reviewed PTX mnemonic.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated `createpolicy` operation to its reviewed PTX
/// instruction.
pub(crate) fn convert_generated_cache_policy(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_instruction: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "generated createpolicy operation requires 1 operand, got {}",
            operands.len()
        );
    }
    let result_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_instruction} $0, $1;"),
        "=l,f",
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
