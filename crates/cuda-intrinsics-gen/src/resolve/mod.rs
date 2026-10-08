/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod abi_ledger;
mod driver;
mod families;
mod guards;
mod materialize;
mod overlay;
mod policy;
mod targets;
#[cfg(test)]
mod tests;

pub(crate) use abi_ledger::validate_operation_key;
pub use driver::resolve;
pub(crate) use families::cluster::cluster_memory_inline_recipe;
pub(crate) use families::mbarrier_extended::mbarrier_extended_inline_recipe;
#[cfg(test)]
pub(crate) use overlay::CATALOG_SCHEMA;
#[allow(unused_imports)]
pub(crate) use targets::{resolve_target_contract, resolve_target_contracts};
