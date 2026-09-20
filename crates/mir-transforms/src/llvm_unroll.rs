/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// --- LLM-generated --- //

//! Forward `#[llvm_unroll]` requests to LLVM, switched on by a
//! `mir.llvm_unroll_hint` operation.
//!
//! This is the companion to [`crate::unroll`], and the contrast is the point.
//! The unroll pass proves a trip count itself and rewrites the loop; when its
//! analysis does not recognize the loop shape it warns and leaves the loop
//! alone. LLVM's unroller has a stronger trip-count analysis (SCEV) and runs
//! anyway as part of `opt -O2`, so `#[llvm_unroll]` asks it to do the work
//! instead.
//!
//! Nothing here rewrites a loop. The pass moves each hint onto the latch(es) of
//! its loop as a [`LlvmLoopUnrollAttr`], where lowering can turn it into
//! `!llvm.loop` metadata on the branch that closes the loop -- which is where
//! LLVM looks for it. Every latch of one loop shares a `group`, because LLVM
//! requires all latches of a loop to reference one metadata node.
//!
//! The request is inert in builds that skip `opt` (`CUDA_OXIDE_NO_OPT=1`, and
//! full variable-debug builds, which also skip this pass entirely).

use dialect_mir::attributes::LlvmLoopUnrollAttr;
use dialect_mir::ops::control_flow::{
    MirCondBranchOp, MirGotoOp, MirLlvmUnrollHintOp, set_llvm_loop_unroll,
};
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::result::Result;

use crate::analyses::loop_info::LoopInfo;

/// Whether to print pass progress notes to stderr.
fn verbose() -> bool {
    std::env::var("CUDA_OXIDE_VERBOSE").is_ok()
}

/// How the author spelled the request, for diagnostics.
fn request_kind(factor: u32) -> String {
    if factor == 0 {
        "#[llvm_unroll]".to_string()
    } else {
        format!("#[llvm_unroll({factor})]")
    }
}

/// Move every `mir.llvm_unroll_hint` onto the latch(es) of its loop, so
/// lowering can emit `!llvm.loop` unroll metadata there.
///
/// Functions without a hint are left byte-for-byte untouched. A hint that does
/// not sit inside a recognizable loop warns and is dropped: the author asked
/// for unrolling, so it is never a silent no-op.
///
/// Run this *after* [`crate::unroll::unroll_annotated_loops`] and its cleanup.
/// That cleanup runs `simplify_cfg`, which may merge or delete blocks; tagging
/// a latch before it could lose the attribute with the block that carried it.
pub fn attach_llvm_loop_metadata(
    module: Ptr<Operation>,
    ctx: &mut Context,
    // Threaded to match pliron's pass shape. Unused for the same reason the
    // unroll pass ignores it: we build a fresh dominator tree below rather than
    // trust a manager whose cache predates that pass's CFG rewrites.
    _analyses: &mut AnalysisManager,
) -> Result<()> {
    // One counter per module: a metadata node is a module-level entity, so
    // groups must not collide across functions.
    let mut next_group = 0u32;

    for func_op in collect_functions(module, ctx) {
        let region = func_op.deref(ctx).get_region(0);
        let hints = collect_hints(ctx, region);
        if hints.is_empty() {
            continue;
        }

        let info = {
            let mut dom_info = DomInfo::default();
            let dom = dom_info.get_dom_tree(ctx, region);
            LoopInfo::compute(ctx, region, dom)
        };

        // Several hints can land in one loop (an annotated loop whose body the
        // frontend split, or two annotations resolving to the same loop). They
        // share one group, so the loop still gets exactly one metadata node.
        let mut groups: Vec<(usize, u32)> = Vec::new();

        for (hint_op, block, factor) in &hints {
            let Some(loop_id) = info.innermost_loop(*block) else {
                eprintln!(
                    "warning: {} requested but no metadata was attached: the annotation is not inside a recognizable loop",
                    request_kind(*factor)
                );
                hint_op.unlink(ctx);
                continue;
            };

            let group = match groups.iter().find(|(id, _)| *id == loop_id) {
                Some((_, group)) => *group,
                None => {
                    let group = next_group;
                    next_group += 1;
                    groups.push((loop_id, group));
                    group
                }
            };

            let latches = info.loops()[loop_id].latches.clone();
            let request = LlvmLoopUnrollAttr {
                factor: *factor,
                group,
            };
            for latch in latches {
                set_latch_request(ctx, latch, request);
            }

            if verbose() {
                eprintln!(
                    "llvm-unroll: loop#{loop_id} factor={factor} group={group} latches={}",
                    info.loops()[loop_id].latches.len()
                );
            }

            hint_op.unlink(ctx);
        }
    }

    Ok(())
}

/// Record `request` on `latch`'s terminator.
///
/// A latch always ends in a branch back to the header, so the terminator is a
/// `mir.goto` or a `mir.cond_br`. Anything else means the CFG is not the shape
/// `LoopInfo` reported; leave it alone rather than tag an op whose lowering
/// would drop the request anyway.
fn set_latch_request(ctx: &mut Context, latch: Ptr<BasicBlock>, request: LlvmLoopUnrollAttr) {
    let Some(terminator) = latch.deref(ctx).get_terminator(ctx) else {
        return;
    };
    let is_branch = Operation::get_op::<MirGotoOp>(terminator, ctx).is_some()
        || Operation::get_op::<MirCondBranchOp>(terminator, ctx).is_some();
    if is_branch {
        set_llvm_loop_unroll(ctx, terminator, request);
    }
}

/// Collect the `mir.func` operations in `module`.
fn collect_functions(module: Ptr<Operation>, ctx: &Context) -> Vec<Ptr<Operation>> {
    let mut out = Vec::new();
    let module_region = module.deref(ctx).get_region(0);
    let blocks: Vec<Ptr<BasicBlock>> = module_region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirFuncOp>(op, ctx).is_some() {
                out.push(op);
            }
        }
    }
    out
}

/// Find the `mir.llvm_unroll_hint` ops in `region`, each with the block it sits
/// in (used to locate the enclosing loop) and its requested factor (0 = full).
fn collect_hints(
    ctx: &Context,
    region: Ptr<Region>,
) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>, u32)> {
    let mut out = Vec::new();
    let blocks: Vec<Ptr<BasicBlock>> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if let Some(hint) = Operation::get_op::<MirLlvmUnrollHintOp>(op, ctx) {
                out.push((op, block, hint.factor(ctx)));
            }
        }
    }
    out
}
