// CCL.cuh
// GPU Connected Components Labeling — iterative parallel union-find.
// Produces per-component stats and, optionally, GPU-filtered char candidates.

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

// ─── Filtered blob (output of filterBlobsKernel) ─────────────────────────────
// POD — written to device memory by GPU, DMA'd to pinned host by the pipeline.
struct FilteredBlob {
    int x, y, width, height, compactId;
};

static constexpr int CCL_MAX_COMPONENTS = 4096;
static constexpr int CCL_MAX_FILTERED   = CCL_MAX_COMPONENTS;

// ─── Pre-allocated workspace (own one per pipeline slot, reuse every frame) ───
// All device memory; pre-allocated in PipelineContext to eliminate hot-path
// cudaMalloc/cudaFree overhead.
struct CCLWorkspace {
    int32_t*        d_label;       // [N] union-find labels
    int32_t*        d_compactMap;  // [N] dense component ID per root pixel
    int*            d_changed;     // [1] convergence flag (device)
    int*            d_numComp;     // [1] component count (device, read on GPU by filter)
    ComponentStats* d_stats;       // [CCL_MAX_COMPONENTS]
};

// Allocate a workspace for an image of N = width * height pixels.
CCLWorkspace allocWorkspace(int N);
void         freeWorkspace(CCLWorkspace& ws);

// ─── Full CCL pipeline + GPU blob filter (preferred API) ──────────────────────
//
// Runs CCL on d_thresh, filters components on GPU, then async-copies survivors
// to pinned host memory so the CPU can read without stalling on a PCIe transfer.
//
// ws             : pre-allocated workspace (allocWorkspace(width*height))
// d_filtered     : device buffer  [CCL_MAX_FILTERED × FilteredBlob] — kernel writes
// d_num_filtered : device int     — kernel writes (zeroed internally, don't pre-zero)
// h_filtered     : pinned host    [CCL_MAX_FILTERED × FilteredBlob] — valid after sync
// h_num_filtered : pinned host int — valid after caller syncs the stream
//
// Fully asynchronous — no host sync inside. Caller must
// cudaStreamSynchronize(stream) (or cudaDeviceSynchronize) before reading
// h_filtered / *h_num_filtered.
//
// allowCoop: pass true only when this is the SOLE kernel running on the device
//   (e.g. sceneStream where no concurrent kernels are active).  Cooperative
//   launch fills every SM simultaneously; launching it from multiple parallel
//   streams causes a grid.sync() deadlock.  Plate streams must leave this false.
void runCCLWithFilter(
    const unsigned char* d_thresh,
    int                  width,
    int                  height,
    CCLWorkspace&        ws,
    FilteredBlob*        d_filtered,
    int*                 d_num_filtered,
    FilteredBlob*        h_filtered,
    int*                 h_num_filtered,
    cudaStream_t         stream,
    int                  maxIter   = 256,
    bool                 allowCoop = false);

// ─── Legacy API (kept for compatibility; always synchronises before returning) ─
int runCCL(const unsigned char* d_thresh,
           int                  width,
           int                  height,
           ComponentStats*      h_statsOut,
           int                  maxIter = 256);
