#!/usr/bin/env bash
# Usage:
#   ./prepare_dataset.sh            # convert + normalize into ppm/
#   ./prepare_dataset.sh --clean    # empty ppm/ first, then regenerate
#
set -euo pipefail

# Resolve paths relative to this script so it runs from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/../license-plate-dataset"
JPGS_DIR="$DATASET_DIR/jpgs"
PPM_DIR="$DATASET_DIR/ppm"

PYTHON="${PYTHON:-python3}"

# --- sanity checks ----------------------------------------------------------
if [[ ! -d "$JPGS_DIR" ]]; then
    echo "ERROR: source images not found at $JPGS_DIR" >&2
    exit 1
fi

if ! "$PYTHON" -c "import PIL" 2>/dev/null; then
    echo "ERROR: Pillow is required.  Install with:  $PYTHON -m pip install Pillow" >&2
    exit 1
fi

# --- optional clean ---------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning $PPM_DIR ..."
    rm -f "$PPM_DIR"/*.ppm
fi

mkdir -p "$PPM_DIR"

# --- step 1: convert to PPM (output lands next to the sources) --------------
echo "==> Step 1/3: converting images in $JPGS_DIR to PPM"
"$PYTHON" "$SCRIPT_DIR/convert_to_ppm.py" "$JPGS_DIR"

# --- step 2: normalize (letterbox) into ppm/ --------------------------------
echo "==> Step 2/3: normalizing PPMs into $PPM_DIR"
"$PYTHON" "$SCRIPT_DIR/normalize_images.py" "$JPGS_DIR" --output-dir "$PPM_DIR"

# --- step 3: clean up intermediate PPMs from jpgs/ --------------------------
echo "==> Step 3/3: removing intermediate PPMs from $JPGS_DIR"
rm -f "$JPGS_DIR"/*.ppm

echo "Done. Normalized PPMs are in: $PPM_DIR"
