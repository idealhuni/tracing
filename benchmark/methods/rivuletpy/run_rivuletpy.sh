#!/usr/bin/env bash
# Run original rivuletpy (R2Tracer) on a single image, output SWC.
#
# Usage: ./run_rivuletpy.sh <input.tif> <output.swc>
# Requires: conda activate tracer-bench  (or tracer-bench env installed)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-tracer-bench}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:?Usage: $0 <input.tif> <output.swc>}"
OUTPUT="${2:?Usage: $0 <input.tif> <output.swc>}"

mkdir -p "$(dirname "$OUTPUT")"

conda run -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/trace_rivuletpy.py" \
    "$INPUT" "$OUTPUT"
