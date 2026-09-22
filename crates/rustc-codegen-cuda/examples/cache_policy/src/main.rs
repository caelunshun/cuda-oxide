/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end example for the `createpolicy` L2 cache-policy intrinsics.
//!
//! One kernel builds all eight fractional policies from a runtime fraction and
//! feeds one of them to a real `cp.async.bulk ... .L2::cache_hint` copy. The
//! policy encoding is opaque, so the host checks that the hinted copy moved
//! the right bytes and that every policy came back non-zero.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_try_wait,
};
use cuda_device::cache_policy::{
    createpolicy_fractional_evict_first, createpolicy_fractional_evict_first_evict_first,
    createpolicy_fractional_evict_last, createpolicy_fractional_evict_last_evict_first,
    createpolicy_fractional_evict_normal, createpolicy_fractional_evict_normal_evict_first,
    createpolicy_fractional_evict_unchanged, createpolicy_fractional_evict_unchanged_evict_first,
};
use cuda_device::tma::cp_async_bulk_g2s_cache_hint;
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const NUM_POLICIES: usize = 8;
const COPY_WORDS: usize = 256;
const COPY_BYTES: u32 = (COPY_WORDS * 4) as u32;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn cache_policy_copy(
        input: &[u32],
        fraction: f32,
        mut policies: DisjointSlice<[u64; NUM_POLICIES]>,
        mut out: DisjointSlice<u32>,
    ) {
        static mut TILE: SharedArray<u32, COPY_WORDS, 128> = SharedArray::UNINIT;
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Streaming input that is read once: mark it evict-first.
        let policy = createpolicy_fractional_evict_first(fraction);

        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, thread::blockDim_x());
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        if tid == 0 {
            unsafe {
                cp_async_bulk_g2s_cache_hint(
                    &raw mut TILE as *mut u8,
                    input.as_ptr() as *const u8,
                    COPY_BYTES,
                    &raw mut BAR,
                    policy,
                );
            }
        }

        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, COPY_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };
        unsafe { while !mbarrier_try_wait(&raw const BAR, token) {} }
        thread::sync_threads();

        let idx = gid.get();
        if idx < COPY_WORDS {
            let value = unsafe { TILE[idx] };
            if let Some(slot) = out.get_mut(gid) {
                *slot = value;
            }
        }

        if idx == 0
            && let Some(row) = policies.get_mut(thread::index_1d())
        {
            row[0] = createpolicy_fractional_evict_last(fraction);
            row[1] = createpolicy_fractional_evict_normal(fraction);
            row[2] = policy;
            row[3] = createpolicy_fractional_evict_unchanged(fraction);
            row[4] = createpolicy_fractional_evict_last_evict_first(fraction);
            row[5] = createpolicy_fractional_evict_normal_evict_first(fraction);
            row[6] = createpolicy_fractional_evict_first_evict_first(fraction);
            row[7] = createpolicy_fractional_evict_unchanged_evict_first(fraction);
        }
    }
}

const LABELS: [&str; NUM_POLICIES] = [
    "evict_last",
    "evict_normal",
    "evict_first",
    "evict_unchanged",
    "evict_last_evict_first",
    "evict_normal_evict_first",
    "evict_first_evict_first",
    "evict_unchanged_evict_first",
];

fn main() {
    println!("=== cache_policy (sm_90+) ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!("skipping: the hinted bulk copy requires sm_90+ (device is sm_{major}{minor})");
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");

    let input: Vec<u32> = (0..COPY_WORDS as u32)
        .map(|i| i.wrapping_mul(0x9e37_79b9))
        .collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut policies_dev = DeviceBuffer::<[u64; NUM_POLICIES]>::zeroed(&stream, 1).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, COPY_WORDS).unwrap();

    // SAFETY: one block of COPY_WORDS threads; every index is in bounds.
    unsafe {
        module.cache_policy_copy(
            &stream,
            LaunchConfig::for_num_elems(COPY_WORDS as u32),
            &input_dev,
            0.5,
            &mut policies_dev,
            &mut out_dev,
        )
    }
    .expect("launch cache_policy_copy");

    let out = out_dev.to_host_vec(&stream).unwrap();
    let policies = policies_dev.to_host_vec(&stream).unwrap()[0];

    let mut passed = true;
    let mismatches = out
        .iter()
        .zip(&input)
        .filter(|(got, want)| got != want)
        .count();
    if mismatches == 0 {
        println!("  hinted bulk copy: ok  ({COPY_WORDS} words)");
    } else {
        println!("  hinted bulk copy: FAIL  {mismatches} of {COPY_WORDS} words differ");
        passed = false;
    }
    for (label, policy) in LABELS.iter().zip(policies) {
        if policy == 0 {
            println!("  {label}: FAIL  policy is zero");
            passed = false;
        } else {
            println!("  {label}: ok  (0x{policy:016x})");
        }
    }

    if !passed {
        println!("FAIL: cache_policy, one or more checks failed");
        std::process::exit(1);
    }
    println!("PASS: {NUM_POLICIES} cache policies built and one consumed on sm_{major}{minor}");
}
