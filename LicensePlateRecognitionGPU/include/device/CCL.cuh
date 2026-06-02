#pragma once
#include <cstdint>
#include <climits>
#include <cuda_runtime.h>

struct ComponentStats {
    int32_t xMin, yMin;
    int32_t xMax, yMax;
    int32_t pixelCount;
    int32_t sumX, sumY;
};

struct FilteredBlob {
    int x, y, width, height, compactId;
};

static constexpr int CCL_MAX_COMPONENTS = 4096;
static constexpr int CCL_MAX_FILTERED   = CCL_MAX_COMPONENTS;

struct CCLWorkspace {
    int32_t*        d_label;       // [N] union-find labels
    int32_t*        d_compactMap;  // [N] dense component ID per root pixel
    int*            d_changed;     // [1] convergence flag (device)
    int*            d_numComp;     // [1] component count (device, read on GPU by filter)
    ComponentStats* d_stats;       // [CCL_MAX_COMPONENTS]
};

CCLWorkspace allocWorkspace(int N);
void         freeWorkspace(CCLWorkspace& ws);

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

int runCCL(const unsigned char* d_thresh,
           int                  width,
           int                  height,
           ComponentStats*      h_statsOut,
           int                  maxIter = 256);
