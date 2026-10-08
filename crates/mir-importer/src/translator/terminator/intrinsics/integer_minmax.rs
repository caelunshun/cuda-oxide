// --- LLM-generated --- //
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler integer min/max intrinsics.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::rust_intrinsics;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

/// Integer min/max intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustIntegerMinMaxIntrinsic {
    /// `core::intrinsics::integer_min`.
    Min,
    /// `core::intrinsics::integer_max`.
    Max,
}

impl RustIntegerMinMaxIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::integer_min" | "std::intrinsics::integer_min" => Some(Self::Min),
            "core::intrinsics::integer_max" | "std::intrinsics::integer_max" => Some(Self::Max),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::Min => rust_intrinsics::CALLEE_INTEGER_MIN,
            Self::Max => rust_intrinsics::CALLEE_INTEGER_MAX,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc integer min/max intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_integer_minmax_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustIntegerMinMaxIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    helpers::emit_function_call(
        ctx,
        body,
        intrinsic.placeholder_callee(),
        args,
        destination,
        return_type,
        None,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_core_path_recognizes_integer_min_max() {
        for (path, expected) in [
            (
                "core::intrinsics::integer_min",
                RustIntegerMinMaxIntrinsic::Min,
            ),
            (
                "std::intrinsics::integer_min",
                RustIntegerMinMaxIntrinsic::Min,
            ),
            (
                "core::intrinsics::integer_max",
                RustIntegerMinMaxIntrinsic::Max,
            ),
            (
                "std::intrinsics::integer_max",
                RustIntegerMinMaxIntrinsic::Max,
            ),
        ] {
            assert_eq!(
                RustIntegerMinMaxIntrinsic::from_core_path(path),
                Some(expected)
            );
        }
        assert_eq!(
            RustIntegerMinMaxIntrinsic::from_core_path("core::cmp::min"),
            None
        );
    }
}
