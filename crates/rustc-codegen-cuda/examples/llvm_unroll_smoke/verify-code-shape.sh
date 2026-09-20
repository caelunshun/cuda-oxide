#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --- LLM-generated --- #

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pre="${root}/llvm_unroll_smoke.ll"
post="${root}/llvm_unroll_smoke.opt.ll"
test -s "${pre}"
test -s "${post}"

python3 - "${pre}" "${post}" <<'PY'
import re
import sys

pre = open(sys.argv[1], encoding="utf-8").read()
post = open(sys.argv[2], encoding="utf-8").read()


def body(ir, name):
    """The text of @name's definition, brace-matched."""
    header = re.search(r"\ndefine [^\n]*@" + re.escape(name) + r"\(", ir)
    if header is None:
        sys.exit(f"error: missing definition of {name}")
    start = ir.index("{", header.end() - 1)
    depth, i = 0, start
    while i < len(ir):
        depth += (ir[i] == "{") - (ir[i] == "}")
        if depth == 0:
            return ir[start:i]
        i += 1
    sys.exit(f"error: unterminated definition of {name}")


# 1. Pre-opt: cuda-oxide emits the request, and only for annotated loops.
for kernel, expected in (("full", True), ("partial", True), ("control", False)):
    tagged = "!llvm.loop" in body(pre, kernel)
    if tagged != expected:
        state = "carries" if tagged else "lacks"
        sys.exit(f"error: {kernel} unexpectedly {state} !llvm.loop metadata before opt")

for node in ('!{!"llvm.loop.unroll.full"}', '!{!"llvm.loop.unroll.count", i32 4}'):
    if node not in pre:
        sys.exit(f"error: missing loop-unroll property node {node}")

# A loop id must be `distinct` and name itself first, or LLVM does not
# recognize it as one.
ids = re.findall(r"^!(\d+) = distinct !\{!(\d+), !\d+\}$", pre, flags=re.M)
if not ids:
    sys.exit("error: no distinct !llvm.loop node was emitted")
for node_id, self_ref in ids:
    if node_id != self_ref:
        sys.exit(f"error: loop node !{node_id} is not self-referential")

# The marker call the macro plants is consumed during import; it must never
# survive as a real function.
if "__llvm_unroll_config" in pre:
    sys.exit("error: the __llvm_unroll_config marker leaked into the LLVM module")

# 2. Post-opt: LLVM acted on the request. Each source iteration writes its
# accumulator volatilely, so body copies are countable.
full = body(post, "full")
full_stores = full.count("store volatile")
if full_stores != 8:
    sys.exit(f"error: full was not fully unrolled: {full_stores} body copies, want 8")
if "!llvm.loop" in full:
    sys.exit("error: full still has a loop after a full unroll")

# `#[llvm_unroll(4)]` means four copies per trip plus a remainder copy. Exactly
# five is also what proves the factor came from the annotation: LLVM's own
# choice for this loop is a different number (see `control` below).
partial_stores = body(post, "partial").count("store volatile")
if partial_stores != 5:
    sys.exit(
        f"error: partial has {partial_stores} body copies, want 5 (4 requested + 1 remainder)"
    )

control_stores = body(post, "control").count("store volatile")
if control_stores == partial_stores:
    sys.exit(
        "error: the unannotated control kernel has the same shape as partial, so this "
        "check no longer proves the unroll factor came from #[llvm_unroll(4)]"
    )

print(f"full: fully unrolled into {full_stores} copies PASS")
print(f"partial: unrolled by 4 into {partial_stores} copies PASS")
print(f"control: unannotated, LLVM chose {control_stores} copies PASS")
PY
