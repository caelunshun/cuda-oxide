/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        DebugKind, NvvmExportConfig, PtxExportConfig, export_module_to_string,
        export_module_to_string_with_config,
    },
    ops::{FuncOp, ReturnOp},
    types::{ArrayType, FuncType, PointerType, VoidType},
};
use pliron::{
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    utils::apint::APInt,
};
use reserved_oxide_symbols::{
    LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX, LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};
use std::num::NonZero;

use crate::common::{DebugConfig, metadata_id, module_top_block, src_location};

#[test]
fn grid_constant_parameter_emits_byval_and_nvvm_annotation() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let pointer_ty = PointerType::get(&ctx, 0);
    let byte_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let descriptor_ty = ArrayType::get(&ctx, byte_ty.into(), 128);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![pointer_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "consume_map".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let alignment = IntegerAttr::new(u64_ty, APInt::from_u64(64, NonZero::new(64).unwrap()));
    {
        let attrs = &mut func.get_operation().deref_mut(&ctx).attributes;
        attrs.set(
            "gpu_kernel".try_into().unwrap(),
            StringAttr::new("true".into()),
        );
        attrs.set(
            format!("{LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX}0")
                .as_str()
                .try_into()
                .unwrap(),
            TypeAttr::new(descriptor_ty.into()),
        );
        attrs.set(
            format!("{LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX}0")
                .as_str()
                .try_into()
                .unwrap(),
            alignment,
        );
    }
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default())
        .expect("NVVM export succeeds");

    assert!(
        ir.contains("define void @consume_map(ptr byval([128 x i8]) align 64 %v0)"),
        "grid-constant pointer must carry the descriptor bytes by value:\n{ir}"
    );
    assert!(ir.contains("!1 = !{i32 1}"), "parameter list:\n{ir}");
    assert!(
        ir.contains("!2 = !{ptr @consume_map, !\"grid_constant\", !1}"),
        "grid-constant annotation:\n{ir}"
    );
    assert!(ir.contains("!nvvm.annotations = !{!0, !2}"));
}

#[test]
fn nvvm_metadata_version_uses_next_allocated_metadata_id() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "bounded_kernel".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let width = NonZero::new(32).unwrap();
    let max_threads = IntegerAttr::new(u32_ty, APInt::from_u32(256, width));
    let min_blocks = IntegerAttr::new(u32_ty, APInt::from_u32(2, width));

    {
        let attrs = &mut func.get_operation().deref_mut(&ctx).attributes;
        attrs.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
        attrs.set(Identifier::try_from("maxntid").unwrap(), max_threads);
        attrs.set(Identifier::try_from("minctasm").unwrap(), min_blocks);
    }

    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default())
        .expect("NVVM export succeeds");

    assert!(
        ir.contains("!0 = !{ptr @bounded_kernel, !\"kernel\", i32 1}"),
        "a launch-bounded kernel still needs its kernel annotation:\n{ir}"
    );
    assert!(
        ir.contains("!nvvm.annotations = !{!0, !1, !2, !3, !4}"),
        "kernel identity plus launch-bounds annotations should occupy !0..!4:\n{ir}"
    );
    assert!(
        ir.contains("!nvvmir.version = !{!5}\n!5 = !{i32 2, i32 0, i32 3, i32 2}"),
        "version metadata should use the next allocated ID:\n{ir}"
    );
}

#[test]
fn export_alwaysinline_function_attribute_uses_llvm_define_syntax() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "inline_helper".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
    func.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(key, StringAttr::new("true".to_string()));
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");
    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define void @inline_helper("))
        .expect("inline helper definition");
    assert_eq!(
        define_line, "define void @inline_helper() alwaysinline #0 {",
        "`alwaysinline` must be emitted after the parameter list, before attr group #0:\n{ir}"
    );
    assert!(
        ir.contains("attributes #0 = { convergent }"),
        "convergent attribute group must still be emitted:\n{ir}"
    );
}

#[test]
fn export_alwaysinline_coexists_with_debug_scope() {
    // alwaysinline and the !dbg scope are emitted on the same define line and
    // must not crowd each other out. This guards the 4-way emission: a future
    // change that drops either one when both are present fails here.
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "inline_helper".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 7, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 8, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    let key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
    func.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(key, StringAttr::new("true".to_string()));
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");
    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define void @inline_helper("))
        .expect("inline helper definition");
    assert!(
        define_line.contains("alwaysinline"),
        "alwaysinline must survive when debug info is on:\n{ir}"
    );
    assert!(
        define_line.contains("!dbg !"),
        "!dbg scope must survive when alwaysinline is present:\n{ir}"
    );
}

#[test]
fn debug_metadata_shares_allocator_with_nvvm_metadata() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    {
        let attrs = &mut func.get_operation().deref_mut(&ctx).attributes;
        attrs.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
    }

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 11, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: NvvmExportConfig::default(),
        debug_kind: DebugKind::LineTables,
    };
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("debug NVVM export succeeds");

    assert!(
        ir.contains("!0 = !DIFile(filename: \"kernel.rs\", directory: \"/tmp/cuda-oxide/tests\")"),
        "debug file node should take the first metadata ID:\n{ir}"
    );
    assert!(
        ir.contains("!4 = !DILocation(line: 11, column: 5, scope: !3)"),
        "instruction location should be allocated before NVVM metadata:\n{ir}"
    );
    assert!(
        ir.contains("!5 = !{ptr @debug_kernel, !\"kernel\", i32 1}"),
        "NVVM annotations should continue after debug metadata:\n{ir}"
    );
    assert!(
        ir.contains("!nvvm.annotations = !{!5}"),
        "named NVVM metadata should reference its allocated node:\n{ir}"
    );
    assert!(
        ir.contains("!nvvmir.version = !{!6}\n!6 = !{i32 2, i32 0, i32 3, i32 2}"),
        "NVVM version should use the next free metadata ID:\n{ir}"
    );
    assert!(
        ir.contains("!llvm.module.flags = !{!7, !8}"),
        "debug module flags should also use the shared allocator:\n{ir}"
    );
}

// --- LLM-generated --- //

/// Build a module with one function whose entry block branches to `dest`, with
/// the branch carrying `request`. Returns the exported LLVM IR.
///
/// A back-edge is not needed: the exporter only looks at the attribute, and
/// keeping the CFG trivial keeps the assertions about metadata alone.
fn export_with_loop_requests(requests: &[llvm_export::ops::LoopUnrollAttr]) -> String {
    use llvm_export::ops::{BrOp, ReturnOp};
    use pliron::basic_block::BasicBlock;

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "loop_unroll".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);

    for (index, request) in requests.iter().enumerate() {
        let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
        let name = format!("latch_{index}");
        let func = FuncOp::new(&mut ctx, name.as_str().try_into().unwrap(), func_ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let region = func.get_operation().deref(&ctx).get_region(0);
        let destination = BasicBlock::new(&mut ctx, None, vec![]);
        destination.insert_at_back(region, &ctx);

        let br = BrOp::new(&mut ctx, destination, vec![]);
        br.get_operation().insert_at_back(entry, &ctx);
        llvm_export::ops::set_loop_unroll(&mut ctx, br.get_operation(), *request);

        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(destination, &ctx);
        func.get_operation().insert_at_back(module_block, &ctx);
    }

    export_module_to_string(&ctx, &module).expect("export succeeds")
}

#[test]
fn full_unroll_request_emits_a_distinct_self_referential_loop_node() {
    let ir = export_with_loop_requests(&[llvm_export::ops::LoopUnrollAttr {
        factor: 0,
        group: 0,
    }]);

    let loop_id = metadata_id(&ir, "distinct !{");
    assert!(
        ir.contains(&format!("br label %bb0, !llvm.loop {loop_id}")),
        "the latch branch must reference its loop node:\n{ir}"
    );

    // LLVM identifies a loop node by its self-reference in operand 0.
    let property_id = metadata_id(&ir, "llvm.loop.unroll.full");
    assert!(
        ir.contains(&format!(
            "{loop_id} = distinct !{{{loop_id}, {property_id}}}"
        )),
        "the loop node must be distinct and name itself first:\n{ir}"
    );
    assert!(
        ir.contains(&format!("{property_id} = !{{!\"llvm.loop.unroll.full\"}}")),
        "a bare request means a full unroll:\n{ir}"
    );
}

#[test]
fn partial_unroll_request_emits_its_count() {
    let ir = export_with_loop_requests(&[llvm_export::ops::LoopUnrollAttr {
        factor: 4,
        group: 0,
    }]);

    let property_id = metadata_id(&ir, "llvm.loop.unroll.count");
    assert!(
        ir.contains(&format!(
            "{property_id} = !{{!\"llvm.loop.unroll.count\", i32 4}}"
        )),
        "a factor must reach LLVM as an unroll count:\n{ir}"
    );
    assert!(
        !ir.contains("llvm.loop.unroll.full"),
        "a counted request is not a full unroll:\n{ir}"
    );
}

#[test]
fn latches_of_one_loop_share_a_node_and_separate_loops_do_not() {
    // Two latches in group 0 (one loop with two back-edges) plus one in group 1.
    let ir = export_with_loop_requests(&[
        llvm_export::ops::LoopUnrollAttr {
            factor: 0,
            group: 0,
        },
        llvm_export::ops::LoopUnrollAttr {
            factor: 0,
            group: 0,
        },
        llvm_export::ops::LoopUnrollAttr {
            factor: 0,
            group: 1,
        },
    ]);

    let references: Vec<&str> = ir
        .lines()
        .filter(|line| line.contains("!llvm.loop"))
        .filter_map(|line| line.split("!llvm.loop ").nth(1))
        .collect();
    assert_eq!(references.len(), 3, "every latch is annotated:\n{ir}");
    assert_eq!(
        references[0], references[1],
        "latches of one loop must share a node:\n{ir}"
    );
    assert_ne!(
        references[0], references[2],
        "separate loops must not share a node:\n{ir}"
    );

    assert_eq!(
        ir.matches("distinct !{").count(),
        2,
        "one loop node per group:\n{ir}"
    );
    assert_eq!(
        ir.matches("llvm.loop.unroll.full").count(),
        1,
        "the identical property node is shared:\n{ir}"
    );
}

#[test]
fn a_branch_without_a_request_is_unchanged() {
    let ir = export_with_loop_requests(&[]);
    assert!(
        !ir.contains("!llvm.loop"),
        "no request means no loop metadata:\n{ir}"
    );
}
