/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// --- LLM-generated --- //

//! Smoke test for `#[llvm_unroll]` / `#[llvm_unroll(N)]`.
//!
//! `#[unroll]` is unrolled by cuda-oxide's own MIR pass, which recognizes
//! explicit counted `while` loops. `#[llvm_unroll]` instead records
//! `!llvm.loop` metadata that LLVM's unroller reads during `opt -O2`, and LLVM
//! derives trip counts with SCEV.
//!
//! The kernels here are deliberately range-based `for` loops -- the shape
//! cuda-oxide's own analysis does not recognize at all, so `#[unroll]` would
//! warn and do nothing. Each iteration writes its accumulator volatilely, so
//! body copies are countable in the generated code, and `verify-code-shape.sh`
//! checks them: `full` collapses into one copy per iteration with no loop left,
//! and `partial` gets exactly the four copies it asked for plus a remainder.
//!
//! Note what `control` shows: LLVM unrolls this loop at `-O2` even unannotated,
//! just by a factor it picks itself (8 here). `#[llvm_unroll(N)]` is how you
//! choose the factor rather than accept that default.
//!
//! Run: cargo oxide run llvm_unroll_smoke

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Iterations of the compile-time-bounded loops.
const TRIP: u32 = 8;

/// `sum of (i & 3) for i in 0..8` == 0+1+2+3+0+1+2+3.
const EXPECTED_SUM: u32 = 12;

#[cuda_module]
mod kernels {
    use super::*;

    /// Range-based `for` with a constant bound, requesting a full LLVM unroll.
    /// cuda-oxide's own pass does not recognize this loop shape; SCEV does.
    /// `out[tid] == 12`.
    #[kernel]
    pub fn full(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            #[llvm_unroll]
            for i in 0..8u32 {
                acc = acc.wrapping_add(i & 3);
                // SAFETY: `out_elem` is this thread's own element.
                unsafe { core::ptr::write_volatile(out_elem, acc) };
            }
        }
    }

    /// The same runtime loop as `partial`, with no annotation: the baseline the
    /// shape check compares against. LLVM leaves one body copy per trip here,
    /// so a passing `partial` cannot be LLVM unrolling it anyway.
    #[kernel]
    pub fn control(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            for i in 0..n {
                acc = acc.wrapping_add(i & 3);
                // SAFETY: `out_elem` is this thread's own element.
                unsafe { core::ptr::write_volatile(out_elem, acc) };
            }
        }
    }

    /// Runtime trip count unrolled by 4. A count request applies even when the
    /// bound is unknown, so this is the form that survives a dynamic `n`.
    /// `out[tid] == sum of (i & 3) for i in 0..n`.
    #[kernel]
    pub fn partial(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            #[llvm_unroll(4)]
            for i in 0..n {
                acc = acc.wrapping_add(i & 3);
                // SAFETY: `out_elem` is this thread's own element.
                unsafe { core::ptr::write_volatile(out_elem, acc) };
            }
        }
    }
}

/// The value `partial` should leave behind for a trip count of `n`.
fn expected_partial(n: u32) -> u32 {
    (0..n).fold(0u32, |acc, i| acc.wrapping_add(i & 3))
}

fn main() {
    println!("=== llvm_unroll_smoke ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_full = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.full(stream.as_ref(), cfg, &mut d_full) }.expect("launch full");
    let got_full = d_full.to_host_vec(&stream).unwrap();

    let mut d_control = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.control(stream.as_ref(), cfg, &mut d_control, 10) }.expect("launch control");
    let got_control = d_control.to_host_vec(&stream).unwrap();

    // A trip count that is not a multiple of 4, so the remainder loop runs too.
    let trip: u32 = 10;
    let mut d_partial = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.partial(stream.as_ref(), cfg, &mut d_partial, trip) }.expect("launch partial");
    let got_partial = d_partial.to_host_vec(&stream).unwrap();

    // A trip count of zero must skip the body entirely, remainder loop included.
    let mut d_empty = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.partial(stream.as_ref(), cfg, &mut d_empty, 0) }.expect("launch partial(0)");
    let got_empty = d_empty.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    for tid in 0..N {
        let checks = [
            ("full", got_full[tid], EXPECTED_SUM),
            ("control", got_control[tid], expected_partial(10)),
            ("partial", got_partial[tid], expected_partial(trip)),
            ("partial(0)", got_empty[tid], 0),
        ];
        for (name, got, want) in checks {
            if got != want {
                println!("FAIL {name} tid={tid}: got {got}, want {want}");
                failures += 1;
            }
        }
    }

    if failures == 0 {
        println!(
            "llvm_unroll_smoke: PASS ({N} threads; full trip {TRIP}, partial trip {trip} by 4)"
        );
    } else {
        println!("llvm_unroll_smoke: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
