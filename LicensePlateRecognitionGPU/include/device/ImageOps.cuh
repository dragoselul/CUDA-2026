// ImageOps.cuh
// GPU image geometry: bilinear resize and affine warp+crop.

#pragma once
#include "Types.h"
#include "KNN.cuh"
#include <cuda_runtime.h>

// ─── Batch warp+crop ──────────────────────────────────────────────────────────
struct WarpParams {
    float cx, cy;
    float angleDeg;
    int   outW, outH;
};

// Rotate d_src around (cx,cy) and crop outW×outH centred on that point.
// All kernel launches are enqueued on `stream` in sequence.
// d_dsts[i] must be pre-allocated to params[i].outW * params[i].outH * 3 bytes.
void runBatchWarpCrop(
    const unsigned char*  d_src,
    int                   W_src,
    int                   H_src,
    const WarpParams*     params,
    unsigned char* const* d_dsts,
    int                   count,
    cudaStream_t          stream = 0);

// ─── Bilinear resize ──────────────────────────────────────────────────────────
// Resize single-channel d_src (srcW×srcH) into pre-allocated d_dst (dstW×dstH).
void runResizeInto(
    const unsigned char* d_src, int srcW, int srcH,
    unsigned char*       d_dst, int dstW, int dstH,
    cudaStream_t         stream = 0);

// ─── Char ROI resize + float conversion ──────────────────────────────────────
// For each of the `numRects` char bounding boxes in h_rects, bilinearly resizes
// the ROI to KNN_CHAR_W × KNN_CHAR_H and writes the result as float32 into
// d_queries[ci * KNN_FEATURES ... (ci+1)*KNN_FEATURES - 1].
//
// h_rects     : host array — uploaded async via cudaMemcpyAsync on `stream`
// d_rects_buf : pre-allocated device buffer [numRects × Rect2i]
// d_queries   : pre-allocated device buffer [numRects × KNN_FEATURES] float32
void runCharROIResize(
    const unsigned char* d_thresh, int W, int H,
    const Rect2i*        h_rects,  int numRects,
    Rect2i*              d_rects_buf,
    float*               d_queries,
    cudaStream_t         stream = 0);
