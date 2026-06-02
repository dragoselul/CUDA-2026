#!/usr/bin/env bash
#
# run.sh
# Build or run the License-Plate GPU pipeline.
#
#   ./run.sh --compile                 # configure + build into build/
#   ./run.sh <path-to-ppm-or-dir>      # run the binary on a .ppm file or a directory
#   ./run.sh                           # run on the default dataset ppm/ directory
#   ./run.sh --compile <path>          # build, then run on <path>
#
# Extra arguments are forwarded straight to the binary, e.g.:
#   ./run.sh ../license-plate-dataset/ppm --max-plates 4 results.csv
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BINARY="$BUILD_DIR/LicensePlateRecognitionGPU"
DEFAULT_PPM_DIR="$SCRIPT_DIR/../license-plate-dataset/ppm"

# --- optional compile step --------------------------------------------------
if [[ "${1:-}" == "--compile" ]]; then
    shift
    echo "==> Configuring (cmake)"
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
    echo "==> Building"
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    echo "Build complete: $BINARY"
    # If no further arguments were given, we only wanted to compile.
    [[ $# -eq 0 ]] && exit 0
fi

# --- run --------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found ($BINARY)." >&2
    echo "       Build it first with:  $0 --compile" >&2
    exit 1
fi

# Default to the dataset's ppm/ directory when no path is supplied.
if [[ $# -eq 0 ]]; then
    echo "==> No path given; using default: $DEFAULT_PPM_DIR"
    set -- "$DEFAULT_PPM_DIR"
fi

echo "==> Running: $BINARY $*"
exec "$BINARY" "$@"
