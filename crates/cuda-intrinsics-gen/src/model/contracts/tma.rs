/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};

/// Closed semantic contract for a TMA operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tma {
    pub operation: TmaOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<TmaReduction>,
    pub adapter: TmaAdapter,
}

impl Tma {
    pub const fn dimensions(&self) -> Option<usize> {
        match &self.reduction {
            Some(reduction) => Some(reduction.dimensions as usize),
            None => self.operation.dimensions(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaOperation {
    G2sTile1d,
    G2sTile2d,
    G2sTile2dMulticast,
    G2sTile2dMulticastCg2,
    G2sTile3d,
    G2sTile4d,
    G2sTile5d,
    S2gTile1d,
    S2gTile2d,
    S2gTile3d,
    S2gTile4d,
    S2gTile5d,
    Reduce,
    BulkG2s,
    BulkG2sCacheHint,
    BulkG2sMulticast,
    BulkG2sMulticastCacheHint,
    BulkG2sCta,
    BulkG2sCtaCacheHint,
    BulkS2g,
    BulkS2gCacheHint,
    BulkS2gByteMask,
    BulkS2gByteMaskCacheHint,
    BulkCtaToCluster,
    #[serde(rename = "bulk_prefetch_l2")]
    BulkPrefetchL2,
    #[serde(rename = "bulk_prefetch_l2_cache_hint")]
    BulkPrefetchL2CacheHint,
    G2sTile1dCacheHint,
    G2sTile2dCacheHint,
    G2sTile2dMulticastCacheHint,
    G2sTile2dMulticastCg2CacheHint,
    G2sTile3dCacheHint,
    G2sTile4dCacheHint,
    G2sTile5dCacheHint,
    S2gTile1dCacheHint,
    S2gTile2dCacheHint,
    S2gTile3dCacheHint,
    S2gTile4dCacheHint,
    S2gTile5dCacheHint,
    CommitGroup,
    WaitGroup,
    WaitGroupRead,
    PrefetchTensorMap,
    PrefetchTile1d,
    PrefetchTile2d,
    PrefetchTile3d,
    PrefetchTile4d,
    PrefetchTile5d,
    #[serde(rename = "prefetch_tile_gather4_2d")]
    PrefetchTileGather4TwoDimensional,
    PrefetchTile1dCacheHint,
    PrefetchTile2dCacheHint,
    PrefetchTile3dCacheHint,
    PrefetchTile4dCacheHint,
    PrefetchTile5dCacheHint,
    #[serde(rename = "prefetch_tile_gather4_2d_cache_hint")]
    PrefetchTileGather4TwoDimensionalCacheHint,
    ReplaceBoxDim,
    ReplaceElementStride,
    ReplaceElementType,
    ReplaceFillMode,
    ReplaceGlobalAddress,
    ReplaceGlobalDim,
    ReplaceGlobalStride,
    ReplaceInterleaveLayout,
    ReplaceRank,
    ReplaceSwizzleAtomicity,
    ReplaceSwizzleMode,
    FenceProxyTensorMapAcquireCluster,
    FenceProxyTensorMapAcquireCta,
    FenceProxyTensorMapAcquireGpu,
    FenceProxyTensorMapAcquireSystem,
    FenceProxyTensorMapReleaseCluster,
    FenceProxyTensorMapReleaseCta,
    FenceProxyTensorMapReleaseGpu,
    FenceProxyTensorMapReleaseSystem,
}

impl TmaOperation {
    pub const fn dimensions(self) -> Option<usize> {
        match self.tensor_copy() {
            Some(copy) => Some(copy.dimensions),
            None => None,
        }
    }

    pub const fn prefetch_coordinate_count(self) -> Option<usize> {
        match self {
            Self::PrefetchTile1d | Self::PrefetchTile1dCacheHint => Some(1),
            Self::PrefetchTile2d | Self::PrefetchTile2dCacheHint => Some(2),
            Self::PrefetchTile3d | Self::PrefetchTile3dCacheHint => Some(3),
            Self::PrefetchTile4d | Self::PrefetchTile4dCacheHint => Some(4),
            Self::PrefetchTile5d
            | Self::PrefetchTile5dCacheHint
            | Self::PrefetchTileGather4TwoDimensional
            | Self::PrefetchTileGather4TwoDimensionalCacheHint => Some(5),
            _ => None,
        }
    }

    /// Return the closed shape of one tiled `cp.async.bulk.tensor` copy.
    pub const fn tensor_copy(self) -> Option<TmaTensorCopy> {
        const fn g2s(
            dimensions: usize,
            multicast: bool,
            cta_group_2: bool,
            cache_hint: bool,
        ) -> Option<TmaTensorCopy> {
            Some(TmaTensorCopy {
                direction: TmaTensorCopyDirection::GlobalToShared,
                dimensions,
                multicast,
                cta_group_2,
                cache_hint,
            })
        }
        const fn s2g(dimensions: usize, cache_hint: bool) -> Option<TmaTensorCopy> {
            Some(TmaTensorCopy {
                direction: TmaTensorCopyDirection::SharedToGlobal,
                dimensions,
                multicast: false,
                cta_group_2: false,
                cache_hint,
            })
        }

        match self {
            Self::G2sTile1d => g2s(1, false, false, false),
            Self::G2sTile2d => g2s(2, false, false, false),
            Self::G2sTile2dMulticast => g2s(2, true, false, false),
            Self::G2sTile2dMulticastCg2 => g2s(2, true, true, false),
            Self::G2sTile3d => g2s(3, false, false, false),
            Self::G2sTile4d => g2s(4, false, false, false),
            Self::G2sTile5d => g2s(5, false, false, false),
            Self::G2sTile1dCacheHint => g2s(1, false, false, true),
            Self::G2sTile2dCacheHint => g2s(2, false, false, true),
            Self::G2sTile2dMulticastCacheHint => g2s(2, true, false, true),
            Self::G2sTile2dMulticastCg2CacheHint => g2s(2, true, true, true),
            Self::G2sTile3dCacheHint => g2s(3, false, false, true),
            Self::G2sTile4dCacheHint => g2s(4, false, false, true),
            Self::G2sTile5dCacheHint => g2s(5, false, false, true),
            Self::S2gTile1d => s2g(1, false),
            Self::S2gTile2d => s2g(2, false),
            Self::S2gTile3d => s2g(3, false),
            Self::S2gTile4d => s2g(4, false),
            Self::S2gTile5d => s2g(5, false),
            Self::S2gTile1dCacheHint => s2g(1, true),
            Self::S2gTile2dCacheHint => s2g(2, true),
            Self::S2gTile3dCacheHint => s2g(3, true),
            Self::S2gTile4dCacheHint => s2g(4, true),
            Self::S2gTile5dCacheHint => s2g(5, true),
            _ => None,
        }
    }

    /// Return the closed shape of one non-tensor `cp.async.bulk` operation.
    pub const fn bulk(self) -> Option<TmaBulk> {
        const fn shape(
            direction: TmaBulkDirection,
            cache_hint: bool,
            multicast: bool,
            byte_mask: bool,
        ) -> Option<TmaBulk> {
            Some(TmaBulk {
                direction,
                cache_hint,
                multicast,
                byte_mask,
            })
        }

        match self {
            Self::BulkG2s => shape(TmaBulkDirection::GlobalToCluster, false, false, false),
            Self::BulkG2sCacheHint => shape(TmaBulkDirection::GlobalToCluster, true, false, false),
            Self::BulkG2sMulticast => shape(TmaBulkDirection::GlobalToCluster, false, true, false),
            Self::BulkG2sMulticastCacheHint => {
                shape(TmaBulkDirection::GlobalToCluster, true, true, false)
            }
            Self::BulkG2sCta => shape(TmaBulkDirection::GlobalToCta, false, false, false),
            Self::BulkG2sCtaCacheHint => shape(TmaBulkDirection::GlobalToCta, true, false, false),
            Self::BulkS2g => shape(TmaBulkDirection::CtaToGlobal, false, false, false),
            Self::BulkS2gCacheHint => shape(TmaBulkDirection::CtaToGlobal, true, false, false),
            Self::BulkS2gByteMask => shape(TmaBulkDirection::CtaToGlobal, false, false, true),
            Self::BulkS2gByteMaskCacheHint => {
                shape(TmaBulkDirection::CtaToGlobal, true, false, true)
            }
            Self::BulkCtaToCluster => shape(TmaBulkDirection::CtaToCluster, false, false, false),
            Self::BulkPrefetchL2 => shape(TmaBulkDirection::PrefetchL2, false, false, false),
            Self::BulkPrefetchL2CacheHint => {
                shape(TmaBulkDirection::PrefetchL2, true, false, false)
            }
            _ => None,
        }
    }

    pub const fn uses_prefetch_cache_hint(self) -> bool {
        matches!(
            self,
            Self::PrefetchTile1dCacheHint
                | Self::PrefetchTile2dCacheHint
                | Self::PrefetchTile3dCacheHint
                | Self::PrefetchTile4dCacheHint
                | Self::PrefetchTile5dCacheHint
                | Self::PrefetchTileGather4TwoDimensionalCacheHint
        )
    }
}

/// The state spaces one tiled `cp.async.bulk.tensor` copy moves between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmaTensorCopyDirection {
    /// `.global` to `.shared::cluster`, completed through an mbarrier.
    GlobalToShared,
    /// `.shared::cta` to `.global`, completed through the bulk async-group.
    SharedToGlobal,
}

/// Closed shape of one tiled `cp.async.bulk.tensor` copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TmaTensorCopy {
    pub direction: TmaTensorCopyDirection,
    pub dimensions: usize,
    pub multicast: bool,
    pub cta_group_2: bool,
    pub cache_hint: bool,
}

impl TmaTensorCopy {
    pub const fn is_g2s(self) -> bool {
        matches!(self.direction, TmaTensorCopyDirection::GlobalToShared)
    }

    pub const fn is_s2g(self) -> bool {
        matches!(self.direction, TmaTensorCopyDirection::SharedToGlobal)
    }
}

/// The state spaces one non-tensor `cp.async.bulk` operation moves between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmaBulkDirection {
    /// `.global` to `.shared::cluster`, completed through an mbarrier.
    GlobalToCluster,
    /// `.global` to `.shared::cta`, completed through an mbarrier.
    GlobalToCta,
    /// `.shared::cta` to `.global`, completed through the bulk async-group.
    CtaToGlobal,
    /// `.shared::cta` to another CTA's `.shared::cluster`, through an mbarrier.
    CtaToCluster,
    /// A hint that prefetches `.global` bytes into L2.
    PrefetchL2,
}

/// Closed shape of one non-tensor `cp.async.bulk` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TmaBulk {
    pub direction: TmaBulkDirection,
    pub cache_hint: bool,
    pub multicast: bool,
    pub byte_mask: bool,
}

impl TmaBulk {
    /// Bulk copies that signal an mbarrier take it as a trailing operand.
    pub const fn uses_barrier(self) -> bool {
        matches!(
            self.direction,
            TmaBulkDirection::GlobalToCluster
                | TmaBulkDirection::GlobalToCta
                | TmaBulkDirection::CtaToCluster
        )
    }

    /// Only the prefetch hint reads one address instead of copying between two.
    pub const fn is_prefetch(self) -> bool {
        matches!(self.direction, TmaBulkDirection::PrefetchL2)
    }
}

/// Closed identity for one TMA tensor-reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaReduction {
    pub operation: TmaReductionOperation,
    pub load_mode: TmaReductionLoadMode,
    pub dimensions: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaReductionOperation {
    Add,
    And,
    Dec,
    Inc,
    Max,
    Min,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaReductionLoadMode {
    Tile,
    Im2col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaAdapter {
    G2sPointersCoordinatesBarrierInjectDefaults,
    G2sPointersCoordinatesBarrierMaskInjectDefaults,
    S2gPointersCoordinatesInjectDefaults,
    ReductionPointersCoordinatesInjectDefaults,
    NoOperands,
    CompileTimeConstantMaxPending,
    DescriptorPointer,
    DescriptorCoordinatesInjectDefaults,
    DescriptorCoordinatesCacheHintInjectFlag,
    DescriptorAndAddressPointers,
    DescriptorOrdinalAndU32,
    DescriptorOrdinalAndU64,
    DescriptorAndImmediateU32,
    DescriptorAndRuntimeU32,
    DescriptorPointerInjectBytes,
    BulkCopyBarrierInjectDefaults,
    BulkCopyBarrierCacheHintInjectFlag,
    BulkCopyBarrierMaskInjectFlag,
    BulkCopyBarrierMaskCacheHintInjectFlags,
    BulkCopyBarrierDirect,
    BulkCopyInjectDefaults,
    BulkCopyCacheHintInjectFlag,
    BulkCopyByteMaskInjectDefaults,
    BulkCopyCacheHintByteMaskInjectFlag,
    BulkPrefetchInjectDefaults,
    BulkPrefetchCacheHintInjectFlag,
    G2sPointersCoordinatesBarrierCacheHintInjectFlag,
    G2sPointersCoordinatesBarrierMaskCacheHintInjectFlags,
    S2gPointersCoordinatesCacheHintInjectFlag,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tma_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "g2s_tile2d_multicast"
adapter = "g2s_pointers_coordinates_barrier_mask_inject_defaults"
"#;
        let parsed = toml::from_str::<Tma>(valid).unwrap();
        assert_eq!(parsed.operation, TmaOperation::G2sTile2dMulticast);
        assert_eq!(
            parsed.adapter,
            TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults
        );

        for invalid in [
            valid.replace("g2s_tile2d_multicast", "g2s_multicast"),
            valid.replace(
                "g2s_pointers_coordinates_barrier_mask_inject_defaults",
                "direct",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Tma>(&invalid).is_err(), "{invalid}");
        }
    }
}
