/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// --- LLM-generated --- //

//! End-to-end tests for the `#[llvm_unroll]` metadata pass.
//!
//! The observable is the opposite of [`unroll.rs`](unroll)'s: this pass must
//! leave the CFG exactly as it found it, and move the request onto the loop's
//! latch(es) where lowering can find it. The loop still being a loop afterwards
//! is the point -- LLVM unrolls it later, cuda-oxide does not.

mod common;

use common::{counted_loop, mir_ctx, multi_latch_counted_loop, nested_counted_loop};
use dialect_mir::attributes::LlvmLoopUnrollAttr;
use dialect_mir::ops::{MirLlvmUnrollHintOp, control_flow::llvm_loop_unroll};
use mir_transforms::analyses::loop_info::LoopInfo;
use mir_transforms::llvm_unroll::attach_llvm_loop_metadata;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::{Operation, verify_operation};
use pliron::pass::AnalysisManager;
use pliron::region::Region;

/// How many natural loops are left in `region`.
fn loop_count(ctx: &Context, region: Ptr<Region>) -> usize {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt).loops().len()
}

/// How many `mir.llvm_unroll_hint` ops remain in `region`.
fn hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    let mut count = 0;
    for block in region.deref(ctx).iter(ctx).collect::<Vec<_>>() {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirLlvmUnrollHintOp>(op, ctx).is_some() {
                count += 1;
            }
        }
    }
    count
}

/// The request recorded on `block`'s terminator, if any.
fn request_on(ctx: &Context, block: Ptr<BasicBlock>) -> Option<LlvmLoopUnrollAttr> {
    let terminator = block.deref(ctx).get_terminator(ctx)?;
    llvm_loop_unroll(ctx, terminator)
}

/// Plant a hint in `block` and run the pass over `module`.
fn tag(ctx: &mut Context, module: Ptr<Operation>, block: Ptr<BasicBlock>, factor: u32) {
    let hint = MirLlvmUnrollHintOp::new(ctx, factor);
    hint.get_operation().insert_at_front(block, ctx);
    verify_operation(module, ctx).expect("input module verifies");

    let mut analyses = AnalysisManager::default();
    attach_llvm_loop_metadata(module, ctx, &mut analyses).expect("llvm-unroll pass succeeds");
    verify_operation(module, ctx).expect("output module verifies");
}

/// A full request moves onto the latch and the loop survives untouched.
#[test]
fn full_request_tags_the_latch_and_keeps_the_loop() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);
    assert_eq!(loop_count(&ctx, lp.region), 1, "starts with one loop");

    tag(&mut ctx, lp.module, lp.latch, 0);

    assert_eq!(
        loop_count(&ctx, lp.region),
        1,
        "this pass must not unroll: LLVM does that later"
    );
    assert_eq!(hint_count(&ctx, lp.region), 0, "the hint is consumed");
    let request = request_on(&ctx, lp.latch).expect("the latch carries the request");
    assert_eq!(request.factor, 0, "factor 0 means a full unroll");
}

/// A partial request records its factor verbatim; `N` reaches LLVM as the
/// `llvm.loop.unroll.count` it will become.
#[test]
fn partial_request_records_its_factor() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    tag(&mut ctx, lp.module, lp.latch, 4);

    let request = request_on(&ctx, lp.latch).expect("the latch carries the request");
    assert_eq!(request.factor, 4);
    assert_eq!(loop_count(&ctx, lp.region), 1, "the loop is left alone");
}

/// Every latch of a multi-latch loop must reference one `!llvm.loop` node, so
/// they all get the same group.
#[test]
fn every_latch_of_one_loop_shares_a_group() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 6, 1, 1);

    tag(&mut ctx, lp.module, lp.normal_latch, 0);

    let normal = request_on(&ctx, lp.normal_latch).expect("normal latch tagged");
    let continue_latch = request_on(&ctx, lp.continue_latch).expect("continue latch tagged");
    assert_eq!(
        normal.group, continue_latch.group,
        "both latches close the same loop, so they share one metadata node"
    );
    assert_eq!(normal.factor, continue_latch.factor);
}

/// Only the annotated loop is tagged. An inner annotation leaves the outer
/// loop's latch alone, exactly as `#[unroll]` only unrolls its own loop.
#[test]
fn only_the_annotated_loop_is_tagged() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    tag(&mut ctx, lp.module, lp.inner_body, 0);

    assert!(
        request_on(&ctx, lp.inner_body).is_some(),
        "the inner loop's latch carries the request"
    );
    assert!(
        request_on(&ctx, lp.outer_latch).is_none(),
        "the outer loop was not annotated and must stay untagged"
    );
    assert_eq!(loop_count(&ctx, lp.region), 2, "both loops remain loops");
}

/// Two annotated loops get distinct groups, so they cannot collapse onto one
/// metadata node.
#[test]
fn separate_loops_get_separate_groups() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    let inner_hint = MirLlvmUnrollHintOp::new(&mut ctx, 0);
    inner_hint
        .get_operation()
        .insert_at_front(lp.inner_body, &ctx);
    let outer_hint = MirLlvmUnrollHintOp::new(&mut ctx, 2);
    outer_hint
        .get_operation()
        .insert_at_front(lp.outer_body, &ctx);

    let mut analyses = AnalysisManager::default();
    attach_llvm_loop_metadata(lp.module, &mut ctx, &mut analyses).expect("pass succeeds");

    let inner = request_on(&ctx, lp.inner_body).expect("inner latch tagged");
    let outer = request_on(&ctx, lp.outer_latch).expect("outer latch tagged");
    assert_ne!(inner.group, outer.group, "one metadata node per loop");
    assert_eq!(inner.factor, 0);
    assert_eq!(outer.factor, 2);
}

/// A hint that is not inside any loop is dropped rather than left to reach
/// lowering. (It also warns; the warning goes to stderr, which the pass owns.)
#[test]
fn a_hint_outside_any_loop_is_dropped() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    tag(&mut ctx, lp.module, lp.preheader, 0);

    assert_eq!(hint_count(&ctx, lp.region), 0, "the stray hint is removed");
    assert!(
        request_on(&ctx, lp.latch).is_none(),
        "a hint outside the loop must not tag the loop"
    );
}

/// Without a hint the pass is a no-op, so an unannotated kernel lowers exactly
/// as it did before this feature existed.
#[test]
fn no_hint_leaves_every_terminator_untagged() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    let mut analyses = AnalysisManager::default();
    attach_llvm_loop_metadata(lp.module, &mut ctx, &mut analyses).expect("pass succeeds");

    for block in [lp.preheader, lp.header, lp.latch, lp.exit] {
        assert!(request_on(&ctx, block).is_none(), "nothing is tagged");
    }
    assert_eq!(loop_count(&ctx, lp.region), 1);
}
