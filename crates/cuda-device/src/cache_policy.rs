/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! L2 cache-eviction policies for `.L2::cache_hint` operands.
//!
//! The `_cache_hint` variants in [`crate::tma`] take an opaque 64-bit policy.
//! PTX does not document how that value is encoded, so the only portable way
//! to build one is `createpolicy` (PTX ISA 7.4, `sm_80+`), exposed here as one
//! function per eviction-priority combination:
//!
//! - `createpolicy_fractional_<primary>(fraction)` applies `<primary>` to
//!   `fraction` of the accessed lines and leaves the rest unchanged.
//! - `createpolicy_fractional_<primary>_evict_first(fraction)` marks the rest
//!   evict-first instead.
//!
//! `fraction` must be in `(0.0, 1.0]`; `1.0` applies the primary priority to
//! every line. The policy is a pure function of its inputs, so it can be
//! computed once and reused across copies.
//!
//! ```rust,ignore
//! use cuda_device::cache_policy::createpolicy_fractional_evict_first;
//! use cuda_device::tma::cp_async_bulk_g2s_cache_hint;
//!
//! // Streaming data that will not be reused: evict it first.
//! let policy = createpolicy_fractional_evict_first(1.0);
//! unsafe { cp_async_bulk_g2s_cache_hint(dst, src, bytes, bar, policy) };
//! ```

include!("generated/cache_policy.rs");
