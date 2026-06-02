# License Plate Recognition — CPU vs GPU

Two implementations of the same license-plate pipeline:

- `LicensePlateRecognitionGPU/` — CUDA implementation (reads `.ppm`).
- `OpenCV_3_License_Plate_Recognition_CPU/` — OpenCV baseline (reads `.jpg`/`.png`).
- `license-plate-dataset/` — shared dataset (`jpgs/` sources, `ppm/` for the GPU build).

Each project has a `run.sh`: `--compile` builds it, otherwise it runs the binary on the path you give.

## Prerequisites

- **GPU:** CMake ≥ 3.18, CUDA Toolkit, an NVIDIA GPU.
- **CPU:** CMake ≥ 3.18, OpenCV.
- **Dataset prep:** Python 3 + Pillow (`pip install Pillow`).

## GPU project

```bash
cd LicensePlateRecognitionGPU

./prepare_dataset.sh          # one-time: convert jpgs/ -> normalized ppm/
./run.sh --compile            # build into build/
./run.sh                      # run on ../license-plate-dataset/ppm
./run.sh <file.ppm|dir>       # run on a specific image or directory
```

Extra args pass through to the binary: `./run.sh <dir> --max-plates 4 out.csv`.

## CPU project

```bash
cd OpenCV_3_License_Plate_Recognition_CPU

./run.sh --compile            # build into build/
./run.sh                      # run on ../license-plate-dataset/jpgs
./run.sh <file|dir|glob>      # run on specific images
```

Extra args pass through: `./run.sh <dir> results.csv summary.csv`.
The CPU version reads `.jpg`/`.png` directly — no dataset prep needed.
