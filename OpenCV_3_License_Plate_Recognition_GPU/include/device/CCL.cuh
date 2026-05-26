// CCL.cuh
// GPU Connected Components Labeling — iterative parallel union-find.
// Produces per-component stats and, optionally, GPU-filtered char candidates
// to reduce PCIe bandwidth (only survivors are transferred to the host).

#pragma once
#include <cstdint>
#include <climits>
#include <cuda_runtime.h>

// ─── Component statistics (internal, computed entirely on GPU) ────────────────
struct ComponentStats {
    int32_t xMin, yMin;
    int32_t xMax, yMax;
    int32_t pixelCount;
    int32_t sumX, sumY;
};

// ─── Filtered blob (output of filterBlobsKernel, transferred to host) ─────────
// POD — safe to write from GPU and copy to pinned host memory.
struct FilteredBlob {
    int x, y, width, height, compactId;
};

static constexpr int CCL_MAX_COMPONENTS = 4096;
static constexpr int CCL_MAX_FILTERED   = CCL_MAX_COMPONENTS;  // worst-case survivors

// ─── Full CCL pipeline + GPU blob filter (preferred API) ──────────────────────
//
// Runs CCL on d_thresh, then immediately filters the per-component stats on
// the GPU (area, size, aspect-ratio checks) so only char-sized blobs cross PCIe.
//
// d_thresh        : binary image on device (0 or 255), W*H bytes
// d_filtered      : pre-allocated device buffer  [CCL_MAX_FILTERED × FilteredBlob]
// d_num_filtered  : pre-allocated device int, must be zeroed before the call
// h_filtered      : pinned host buffer            [CCL_MAX_FILTERED × FilteredBlob]
// h_num_filtered  : pinned host int
//
// The function enqueues an async D2H transfer on `stream` and returns WITHOUT
// synchronising.  The caller is responsible for syncing before reading h_filtered.
void runCCLWithFilter(
    const unsigned char* d_thresh,
    int                  width,
    int                  height,
    FilteredBlob*        d_filtered,
    int*                 d_num_filtered,
    FilteredBlob*        h_filtered,
    int*                 h_num_filtered,
    cudaStream_t         stream,
    int                  maxIter = 256);

// ─── Legacy API (kept for compatibility; always synchronises before returning) ─
int runCCL(const unsigned char* d_thresh,
           int                  width,
           int                  height,
           ComponentStats*      h_statsOut,
           int                  maxIter = 256);
