/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end example for the `tcgen05.ld.red` reducing tensor-memory loads.
//!
//! One warp writes a known pattern into tensor memory with `tcgen05.st`, then
//! reads it back through `tcgen05.ld.red` in its u32, s32, and f32 forms. Each
//! load returns the lane's registers together with their min/max, so the host
//! checks both the loaded values and every reduction against a CPU model.
//!
//! `tcgen05.ld.red` exists only on sm_103a and sm_110a; on any other GPU the
//! example stops after reporting that the PTX was generated.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_dealloc, tcgen05_ld_red_16x32bx2_x2_max_u32,
    tcgen05_ld_red_16x32bx2_x128_min_abs_nan_f32, tcgen05_ld_red_32x32b_x4_max_abs_f32,
    tcgen05_ld_red_32x32b_x4_max_s32, tcgen05_ld_red_32x32b_x4_min_nan_f32,
    tcgen05_ld_red_32x32b_x4_min_u32, tcgen05_ld_red_32x32b_x128_max_f32, tcgen05_load_wait,
    tcgen05_relinquish_alloc_permit, tcgen05_st_32x32b_x4_raw, tcgen05_store_wait,
};
use cuda_device::{CuSimd, DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

const LANES: usize = 32;
const COLUMNS: usize = 4;
/// Per lane: the four loaded u32 registers, then the u32 min, s32 max, f32
/// max-abs, and NaN-propagating f32 min reductions.
const WORDS_PER_LANE: usize = COLUMNS + 4;
/// This lane stores a NaN in one f32 column.
const NAN_LANE: usize = 7;

fn u32_value(lane: usize, column: usize) -> u32 {
    ((lane * 7 + column * 13) % 29 + 1) as u32
}

fn s32_value(lane: usize, column: usize) -> i32 {
    (lane as i32 - 16) * (column as i32 + 1) * if column.is_multiple_of(2) { 1 } else { -1 }
}

fn f32_value(lane: usize, column: usize) -> f32 {
    if lane == NAN_LANE && column == 2 {
        f32::NAN
    } else {
        (lane as f32 - 15.5)
            * (column as f32 + 0.5)
            * if column.is_multiple_of(2) { 1.0 } else { -1.0 }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn tcgen05_ld_red_test(mut output: DisjointSlice<u32>) {
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        unsafe {
            let lane = thread::threadIdx_x() as usize;
            if warp::warp_id() == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 32);
                tcgen05_relinquish_alloc_permit();
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDR as *const u32);

            // Columns 0..4 hold u32 data, 4..8 the s32 bit patterns, and
            // 8..12 the f32 bit patterns; each lane owns its TMEM lane.
            let unsigned = CuSimd::new([
                u32_value(lane, 0),
                u32_value(lane, 1),
                u32_value(lane, 2),
                u32_value(lane, 3),
            ]);
            let signed = CuSimd::new([
                s32_value(lane, 0) as u32,
                s32_value(lane, 1) as u32,
                s32_value(lane, 2) as u32,
                s32_value(lane, 3) as u32,
            ]);
            let float = CuSimd::new([
                f32_value(lane, 0).to_bits(),
                f32_value(lane, 1).to_bits(),
                f32_value(lane, 2).to_bits(),
                f32_value(lane, 3).to_bits(),
            ]);
            tcgen05_st_32x32b_x4_raw(tmem, unsigned);
            tcgen05_st_32x32b_x4_raw(tmem + 4, signed);
            tcgen05_st_32x32b_x4_raw(tmem + 8, float);
            tcgen05_store_wait();

            let (registers, min_u32) = tcgen05_ld_red_32x32b_x4_min_u32(tmem);
            let (_, max_s32) = tcgen05_ld_red_32x32b_x4_max_s32(tmem + 4);
            let (_, max_abs_f32) = tcgen05_ld_red_32x32b_x4_max_abs_f32(tmem + 8);
            let (_, min_nan_f32) = tcgen05_ld_red_32x32b_x4_min_nan_f32(tmem + 8);
            tcgen05_load_wait();

            let base = lane * WORDS_PER_LANE;
            for column in 0..COLUMNS {
                *output.get_unchecked_mut(base + column) = registers[column];
            }
            *output.get_unchecked_mut(base + COLUMNS) = min_u32;
            *output.get_unchecked_mut(base + COLUMNS + 1) = max_s32 as u32;
            *output.get_unchecked_mut(base + COLUMNS + 2) = max_abs_f32.to_bits();
            *output.get_unchecked_mut(base + COLUMNS + 3) = min_nan_f32.to_bits();

            thread::sync_threads();
            if warp::warp_id() == 0 {
                tcgen05_dealloc(tmem, 32);
            }
        }
    }

    /// Keeps the half-split and 128-register reducing forms in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_ld_red(tmem: u32, mut output: DisjointSlice<u32>) {
        unsafe {
            let (split, split_max) = tcgen05_ld_red_16x32bx2_x2_max_u32::<16>(tmem);
            let (wide, wide_max) = tcgen05_ld_red_32x32b_x128_max_f32(tmem);
            let (split_wide, split_wide_min) =
                tcgen05_ld_red_16x32bx2_x128_min_abs_nan_f32::<32>(tmem);
            tcgen05_load_wait();
            let lane = thread::threadIdx_x() as usize;
            *output.get_unchecked_mut(lane) = split[1] ^ split_max;
            *output.get_unchecked_mut(lane + LANES) = (wide[127] + wide_max).to_bits();
            *output.get_unchecked_mut(lane + 2 * LANES) =
                (split_wide[0] + split_wide_min).to_bits();
        }
    }
}

fn can_execute_tcgen05_ld_red(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 3) | (11, 0))
}

fn main() {
    println!("=== tcgen05_ld_red (sm_103a / sm_110a) ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if !can_execute_tcgen05_ld_red(major, minor) {
        println!(
            "skipping: tcgen05.ld.red requires sm_103a or sm_110a (device is sm_{major}{minor}); PTX was generated successfully"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, LANES * WORDS_PER_LANE).unwrap();
    let cfg = LaunchConfig {
        block_dim: (LANES as u32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one warp; every output index is below LANES * WORDS_PER_LANE.
    unsafe { module.tcgen05_ld_red_test(&stream, cfg, &mut output) }
        .expect("launch tcgen05_ld_red_test");
    let output = output.to_host_vec(&stream).unwrap();

    let mut failures = 0;
    for lane in 0..LANES {
        let row = &output[lane * WORDS_PER_LANE..(lane + 1) * WORDS_PER_LANE];
        let unsigned: Vec<u32> = (0..COLUMNS).map(|column| u32_value(lane, column)).collect();
        let signed: Vec<i32> = (0..COLUMNS).map(|column| s32_value(lane, column)).collect();
        let float: Vec<f32> = (0..COLUMNS).map(|column| f32_value(lane, column)).collect();

        let expected_max_abs = float
            .iter()
            .filter(|value| !value.is_nan())
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let expected_min_nan = if float.iter().any(|value| value.is_nan()) {
            None
        } else {
            Some(float.iter().copied().fold(f32::INFINITY, f32::min))
        };
        let min_nan = f32::from_bits(row[COLUMNS + 3]);

        let checks = [
            ("registers", row[..COLUMNS] == unsigned[..]),
            ("min.u32", row[COLUMNS] == *unsigned.iter().min().unwrap()),
            (
                "max.s32",
                row[COLUMNS + 1] as i32 == *signed.iter().max().unwrap(),
            ),
            (
                "max.abs.f32",
                f32::from_bits(row[COLUMNS + 2]) == expected_max_abs,
            ),
            (
                "min.NaN.f32",
                match expected_min_nan {
                    None => min_nan.is_nan(),
                    Some(expected) => min_nan == expected,
                },
            ),
        ];
        for (label, ok) in checks {
            if !ok {
                println!("  lane {lane}: FAIL {label} (row {row:?})");
                failures += 1;
            }
        }
    }

    if failures != 0 {
        println!("FAIL: tcgen05_ld_red, {failures} mismatches");
        std::process::exit(1);
    }
    println!("PASS: tcgen05.ld.red u32/s32/f32 reductions matched on sm_{major}{minor}");
}
