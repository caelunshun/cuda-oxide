#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --- LLM-generated --- #

# Prove that `#[cfg(cuda_arch_min = "80")]` selected the arm the built target
# calls for -- and, below sm_80, that the other arm was *never compiled at
# all* rather than compiled and then discarded.
#
# The script reads the architecture out of the PTX instead of taking it as an
# argument, so it checks whatever build is on disk: `--arch sm_75`,
# `--arch sm_86`, or an auto-detected `cargo oxide run`.
#
# It assumes the build had an architecture, which is what makes "the PTX target
# implies the arm" a valid rule. A build with no architecture configured sets no
# `cuda_arch*` cfgs at all, so it compiles the fallback arm while the backend
# still defaults to sm_80 -- a legitimate combination this check would read as a
# failure. `scripts/smoketest.sh` therefore drives this example through pinned
# builds only.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/arch_cfg.ptx"
ll="${root}/arch_cfg.ll"
test -s "${ptx}"
test -s "${ll}"

target="$(awk '/^\.target /{ print $2; exit }' "${ptx}")"
if [[ -z "${target}" ]]; then
    echo "error: ${ptx} carries no .target line" >&2
    exit 1
fi
# `sm_90a` and `sm_100f` carry a family suffix; the cfg comparison is numeric.
capability="${target#sm_}"
capability="${capability%[af]}"
if ! [[ "${capability}" =~ ^[0-9]+$ ]]; then
    echo "error: cannot read a compute capability out of .target ${target}" >&2
    exit 1
fi

entry="$(awk '/^\.visible \.entry warp_sum\(/,/^}/' "${ptx}")"
if [[ -z "${entry}" ]]; then
    echo "error: ${ptx} has no warp_sum entry" >&2
    exit 1
fi

redux_count="$(grep -cE 'redux\.sync\.add' <<<"${entry}" || true)"
shuffle_count="$(grep -cE 'shfl\.sync\.bfly' <<<"${entry}" || true)"
# The kernel writes `cfg!(cuda_arch_min = "80")` into the last output slot, so
# the immediate in the generated store is the compiled arm, observable without
# a GPU.
marker_one="$(grep -cE 'st\.global\.b32[[:space:]]+\[[^]]*\], 1;' <<<"${entry}" || true)"
marker_zero="$(grep -cE 'st\.global\.b32[[:space:]]+\[[^]]*\], 0;' <<<"${entry}" || true)"

if [[ ${capability} -ge 80 ]]; then
    if [[ ${redux_count} -ne 1 ]]; then
        echo "error: ${target} must lower the single-instruction reduction; found ${redux_count} redux.sync.add." >&2
        echo "       Either \`cuda_arch_min = \"80\"\` did not reach rustc, or this build pinned no" >&2
        echo "       architecture and only the backend default reached sm_80." >&2
        exit 1
    fi
    if [[ ${shuffle_count} -ne 0 ]]; then
        echo "error: ${target} still contains ${shuffle_count} shfl.sync.bfly from the pre-sm_80 fallback" >&2
        exit 1
    fi
    if [[ ${marker_one} -ne 1 || ${marker_zero} -ne 0 ]]; then
        echo "error: ${target} must report the redux arm (cfg! marker store of 1)" >&2
        exit 1
    fi
    echo "arch_cfg code shape (${target}): redux.sync.add selected PASS"
    exit 0
fi

if [[ ${shuffle_count} -ne 5 ]]; then
    echo "error: ${target} must lower the 5-round butterfly; found ${shuffle_count} shfl.sync.bfly" >&2
    exit 1
fi
if [[ ${redux_count} -ne 0 ]]; then
    echo "error: ${target} contains redux.sync.add, which does not exist below sm_80" >&2
    exit 1
fi
if [[ ${marker_zero} -ne 1 || ${marker_one} -ne 0 ]]; then
    echo "error: ${target} must report the shuffle arm (cfg! marker store of 0)" >&2
    exit 1
fi

# The real claim of the whole feature: rustc dropped the dead arm before MIR
# existed, so the collector never walked into `redux_sync_add` and the
# intrinsic never reached LLVM. A backend that merely dead-stripped it later
# would still leave the declaration here.
if grep -q 'llvm[.$]nvvm[.$]redux' "${ll}"; then
    echo "error: ${target} LLVM module mentions llvm.nvvm.redux; the sm_80 arm was compiled and only later removed" >&2
    exit 1
fi

echo "arch_cfg code shape (${target}): shuffle fallback selected, sm_80 arm never compiled PASS"
