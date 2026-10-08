/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::extract::{IMPORTED_SCHEMA, read_upstream_lock};
use crate::model::{
    AbiLedgerFile, CatalogFile, CatalogInputs, CatalogSource, ImportedFile, ImportedIntrinsic,
    IntrinsicSource, OverlayFile, OverlayIntrinsic,
};
use crate::util::{read_json, sha256_file};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::abi_ledger::*;
use super::families::*;
use super::materialize::*;
use super::overlay::*;
use super::policy::*;

pub(super) struct ResolutionBase {
    pub(super) overlay: OverlayFile,
    pub(super) imported: ImportedFile,
    pub(super) source: CatalogSource,
    pub(super) imported_sha256: String,
    pub(super) overlay_sha256: String,
    pub(super) abi_ledger_sha256: String,
}

pub fn resolve(repo_root: &Path) -> Result<CatalogFile> {
    let base = load_resolution_base(repo_root)?;
    let ResolutionBase {
        overlay,
        imported,
        source,
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256,
    } = base;
    let imported_by_record = index_imported_intrinsics(&imported)?;

    let mut intrinsics = Vec::with_capacity(overlay.intrinsics.len());
    for policy in &overlay.intrinsics {
        let source = resolve_policy_source(policy)?;
        let declaration = resolve_imported_declaration(policy, &source, &imported_by_record)?;
        validate_special_register_llvm_exclusion(policy, &imported_by_record)?;
        validate_policy(policy, &source, declaration, overlay.intrinsic_abi)?;
        let backend_lowerings = resolve_backend_lowerings(policy)?;
        intrinsics.push(materialize_record(
            policy,
            source,
            declaration,
            backend_lowerings,
            overlay.intrinsic_abi,
        )?);
    }

    Ok(CatalogFile {
        schema: CATALOG_SCHEMA,
        catalog_version: overlay.catalog_version,
        intrinsic_abi: overlay.intrinsic_abi,
        generator_version: env!("CARGO_PKG_VERSION").to_owned(),
        source,
        inputs: CatalogInputs {
            imported_sha256,
            overlay_sha256,
            abi_ledger_sha256,
        },
        intrinsics,
    })
}

pub(super) fn load_resolution_base(repo_root: &Path) -> Result<ResolutionBase> {
    let lock = read_upstream_lock(repo_root)?;
    let imported_path = repo_root.join("intrinsics/imported.json");
    let overlay_path = repo_root.join("intrinsics/overlay.toml");
    let imported: ImportedFile = read_json(&imported_path)?;
    let (mut overlay, overlay_sha256) = read_overlay(repo_root, &overlay_path)?;
    let ledger_path = repo_root.join(format!("intrinsics/abi-v{}.toml", overlay.intrinsic_abi));
    let ledger_text = fs::read_to_string(&ledger_path)
        .with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: AbiLedgerFile =
        toml::from_str(&ledger_text).with_context(|| format!("parse {}", ledger_path.display()))?;

    ensure!(
        imported.schema == IMPORTED_SCHEMA,
        "unsupported imported.json schema {}",
        imported.schema
    );
    ensure!(
        overlay.schema == OVERLAY_SCHEMA,
        "unsupported overlay.toml schema {}",
        overlay.schema
    );
    ensure!(
        overlay.intrinsic_abi > 0,
        "intrinsic_abi must be a positive integer"
    );
    ensure!(
        imported.source.llvm_revision == lock.llvm.revision,
        "imported facts use LLVM {}, but upstream.lock pins {}",
        imported.source.llvm_revision,
        lock.llvm.revision
    );
    ensure!(
        imported.source.llvm_tblgen_source_revision == lock.llvm.revision,
        "imported facts were not produced by llvm-tblgen built from the pinned source"
    );
    ensure!(
        imported.source.llvm_tblgen_version == lock.llvm_tblgen.version_line,
        "imported facts use llvm-tblgen {:?}, but upstream.lock pins {:?}",
        imported.source.llvm_tblgen_version,
        lock.llvm_tblgen.version_line
    );
    ensure!(
        imported.source.intrinsics_json_sha256 == lock.dumps.intrinsics_sha256,
        "imported intrinsic dump hash does not match upstream.lock"
    );
    ensure!(
        imported.source.nvptx_json_sha256 == lock.dumps.nvptx_sha256,
        "imported NVPTX dump hash does not match upstream.lock"
    );
    let imported_sha256 = sha256_file(&imported_path)?;
    ensure!(
        imported_sha256 == lock.dumps.normalized_imported_sha256,
        "normalized imported.json hash mismatch: upstream.lock records {}, found {}; regenerate from the pinned dumps, and refresh the lock explicitly only for an intentional normalizer change",
        lock.dumps.normalized_imported_sha256,
        imported_sha256
    );

    bind_generated_abi_ids(&mut overlay, &ledger)?;
    overlay
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    validate_execution_control_family_completeness(&overlay.intrinsics)?;
    validate_unique_overlay(&overlay.intrinsics, overlay.intrinsic_abi)?;
    validate_abi_ledger(&overlay, &ledger)?;
    Ok(ResolutionBase {
        overlay,
        imported,
        source: CatalogSource {
            llvm_repository: lock.llvm.repository,
            llvm_revision: lock.llvm.revision,
            llvm_tblgen_version: lock.llvm_tblgen.version_line,
            llvm_tblgen_source_revision: lock
                .llvm_tblgen
                .built_from_llvm_revision
                .context("pinned llvm-tblgen has no source revision")?,
        },
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256: sha256_file(&ledger_path)?,
    })
}

pub(super) fn index_imported_intrinsics(
    imported: &ImportedFile,
) -> Result<BTreeMap<&str, &ImportedIntrinsic>> {
    let imported_by_record: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|intrinsic| (intrinsic.source_record.as_str(), intrinsic))
        .collect();
    ensure!(
        imported_by_record.len() == imported.intrinsics.len(),
        "imported.json contains duplicate source records"
    );
    Ok(imported_by_record)
}

pub(super) fn resolve_imported_declaration<'a>(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    imported_by_record: &'a BTreeMap<&str, &'a ImportedIntrinsic>,
) -> Result<Option<&'a ImportedIntrinsic>> {
    match source {
        IntrinsicSource::LlvmImported { source_record } => Ok(Some(
            *imported_by_record
                .get(source_record.as_str())
                .with_context(|| {
                    format!(
                        "overlay intrinsic {} references missing imported record {}",
                        policy.id, source_record
                    )
                })?,
        )),
        IntrinsicSource::PtxNative { .. } => Ok(None),
    }
}
