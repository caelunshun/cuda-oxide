/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// --- LLM-generated --- //

//! Architecture-conditional device code with `#[cfg(cuda_arch_min = ..)]`.
//!
//! `cargo oxide` resolves one target architecture and injects it into rustc as
//! ordinary `--cfg` flags, so device code selects an implementation the same
//! way portable Rust selects one per platform:
//!
//! ```text
//! --arch sm_86  -->  cuda_arch_min = "80" set  -->  redux.sync.add.u32   (one instruction)
//! --arch sm_75  -->  cuda_arch_min = "80" unset -->  5x shfl.sync.bfly + add (butterfly)
//! ```
//!
//! Both kernels compute the same warp-wide sum. The difference is that
//! `redux.sync` does not exist below sm_80, so on Turing the first arm is not
//! merely slower -- it cannot be lowered at all. rustc evaluates `#[cfg]`
//! before MIR exists, so on an sm_75 build the `redux_sync_add` call is never
//! collected and never reaches the backend; `verify-code-shape.sh` proves that
//! by checking the generated `.ll` as well as the `.ptx`.
//!
//! There is no `ctx.compute_capability()` check anywhere in this file. That is
//! the point: the arch decision happened at build time, so the host does not
//! re-derive it at run time. The kernel instead reports which arm was compiled
//! (via `cfg!`) in the last output slot, and the host prints it.
//!
//! Run:
//!   cargo oxide build arch_cfg --arch sm_75
//!   cargo oxide build arch_cfg --arch sm_86
//!   cargo oxide run arch_cfg

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

/// Every lane participates.
const FULL_MASK: u32 = 0xffff_ffff;

/// Warps per block; also the number of sums the kernel writes.
const WARPS: usize = 4;

/// One block of `WARPS` full warps.
const BLOCK: u32 = (WARPS as u32) * 32;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum each warp's values and have lane 0 write the result.
    ///
    /// `out` is `WARPS + 1` long: one slot per warp, then a trailing slot
    /// holding `1` when the sm_80+ arm was compiled and `0` otherwise.
    #[kernel]
    pub fn warp_sum(data: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();
        let warps = out.len() - 1;

        let value = if gid.get() < warps * 32 {
            data[gid.get()]
        } else {
            0
        };

        // sm_80 added `redux.sync`, a single instruction for the whole warp.
        #[cfg(cuda_arch_min = "80")]
        let total = warp::redux_sync_add(FULL_MASK, value);

        // Below sm_80 the same reduction is a butterfly: five exchange/add
        // rounds, each halving the distance between partners. Every lane ends
        // up with the full sum, matching `redux.sync.add`'s broadcast.
        #[cfg(not(cuda_arch_min = "80"))]
        let total = {
            let mut acc = value;
            let mut partner_distance = 16u32;
            while partner_distance > 0 {
                acc = acc.wrapping_add(warp::shuffle_xor_sync(FULL_MASK, acc, partner_distance));
                partner_distance >>= 1;
            }
            acc
        };

        if lane == 0 {
            let warp_index = gid.get() / 32;
            if warp_index < warps {
                // SAFETY: bounds checked against the reported warp count.
                unsafe {
                    *out.get_unchecked_mut(warp_index) = total;
                }
            }
        }

        // Report the arm the build selected, so the host verifies the actual
        // compiled shape rather than assuming it from the target it asked for.
        if gid.get() == 0 {
            // SAFETY: `warps` is the last valid index of `out` by construction.
            unsafe {
                *out.get_unchecked_mut(warps) = cfg!(cuda_arch_min = "80") as u32;
            }
        }
    }
}

fn main() {
    println!("=== arch_cfg: #[cfg(cuda_arch_min = \"80\")] ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    // Lane `i` of warp `w` contributes `w * 100 + i`, so each warp's sum is
    // distinct and a cross-warp mix-up cannot pass unnoticed.
    let host_data: Vec<u32> = (0..BLOCK)
        .map(|tid| (tid / 32) * 100 + (tid % 32))
        .collect();
    let expected: Vec<u32> = (0..WARPS as u32)
        .map(|warp_index| (0..32).map(|lane| warp_index * 100 + lane).sum())
        .collect();

    let d_data = DeviceBuffer::from_host(&stream, &host_data).unwrap();
    let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, WARPS + 1).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape/resources match the kernel; both buffers cover
    // every access the kernel makes.
    unsafe { module.warp_sum(stream.as_ref(), cfg, &d_data, &mut d_out) }.expect("launch warp_sum");
    let got = d_out.to_host_vec(&stream).unwrap();

    let took_redux_path = got[WARPS] == 1;
    let path = if took_redux_path {
        "redux path"
    } else {
        "shuffle fallback"
    };

    let mut failures = 0usize;
    for (warp_index, want) in expected.iter().enumerate() {
        if got[warp_index] != *want {
            println!(
                "FAIL warp {warp_index}: got {}, want {want}",
                got[warp_index]
            );
            failures += 1;
        }
    }
    // The compiled arm must be one of the two, not some third value left over
    // from an unwritten slot.
    if got[WARPS] > 1 {
        println!("FAIL arm marker: got {}, want 0 or 1", got[WARPS]);
        failures += 1;
    }

    if failures == 0 {
        println!("arch_cfg: PASS ({path})");
    } else {
        println!("arch_cfg: FAIL ({failures} mismatches, {path})");
        std::process::exit(1);
    }
}
