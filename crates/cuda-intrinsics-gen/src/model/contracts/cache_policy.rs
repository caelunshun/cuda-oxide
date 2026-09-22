/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};

/// Closed identity for one `createpolicy` L2 cache-eviction policy form.
///
/// The produced 64-bit value is opaque: PTX does not document its encoding,
/// so the only reviewed way to obtain one is the instruction itself. The
/// value feeds the `.L2::cache_hint` operand of the bulk-copy, TMA, and
/// prefetch families.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePolicy {
    pub form: CachePolicyForm,
    pub primary: CachePolicyPriority,
    pub secondary: CachePolicySecondaryPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicyForm {
    /// `createpolicy.fractional`: the primary priority applies to a fraction
    /// of the accessed lines, the secondary priority to the rest.
    Fractional,
}

/// Eviction priority applied to the primary share of the accessed lines.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicyPriority {
    EvictLast,
    EvictNormal,
    EvictFirst,
    EvictUnchanged,
}

/// Eviction priority applied outside the primary share. PTX defaults the
/// omitted qualifier to `evict_unchanged`, so only `evict_first` is spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicySecondaryPriority {
    EvictUnchanged,
    EvictFirst,
}

impl CachePolicyPriority {
    pub const fn ptx_qualifier(self) -> &'static str {
        match self {
            Self::EvictLast => "L2::evict_last",
            Self::EvictNormal => "L2::evict_normal",
            Self::EvictFirst => "L2::evict_first",
            Self::EvictUnchanged => "L2::evict_unchanged",
        }
    }
}

impl CachePolicy {
    /// The exact PTX instruction, without operands.
    pub fn ptx_instruction(&self) -> String {
        let CachePolicyForm::Fractional = self.form;
        match self.secondary {
            CachePolicySecondaryPriority::EvictUnchanged => {
                format!(
                    "createpolicy.fractional.{}.b64",
                    self.primary.ptx_qualifier()
                )
            }
            CachePolicySecondaryPriority::EvictFirst => format!(
                "createpolicy.fractional.{}.L2::evict_first.b64",
                self.primary.ptx_qualifier()
            ),
        }
    }
}
