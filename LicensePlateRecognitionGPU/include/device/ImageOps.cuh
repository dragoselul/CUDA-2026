#pragma once
#include "Types.h"
#include "KNN.cuh"
#include <cuda_runtime.h>

struct WarpParams {
    float cx, cy;
    float angleDeg;
    int   outW, outH;
};

void runBatchWarpCrop(
    const unsigned char*  d_src,
    int                   W_src,
    int                   H_src,
    const WarpParams*     params,
    unsigned char* const* d_dsts,
    int                   count,
    cudaStream_t          stream = 0);

void runResizeInto(
    const unsigned char* d_src, int srcW, int srcH,
    unsigned char*       d_dst, int dstW, int dstH,
    cudaStream_t         stream = 0);

void runCharROIResize(
    const unsigned char* d_thresh, int W, int H,
    const Rect2i*        h_rects,  int numRects,
    Rect2i*              d_rects_buf,
    float*               d_queries,
    cudaStream_t         stream = 0);
