#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ─── Intermediate GPU buffers for the 6-kernel pipeline ──────────────────────
// The caller allocates these once (via allocPreprocessBuffers) and reuses
// them across calls.  preprocessDevice does NOT free or sync — it only
// enqueues work on `stream`, so multiple plate streams can run concurrently.
struct PreprocessBuffers {
    unsigned char* d_value;       // BGR → V (max channel)
    unsigned char* d_eroded;
    unsigned char* d_opened;      // dilate(erode(V)) = opening
    unsigned char* d_dilated;
    unsigned char* d_closed;      // erode(dilate(V)) = closing
    unsigned char* d_contrasted;
    unsigned char* d_blurred;
};

// Allocate seven single-channel buffers for an image of the given size.
// All buffers are W*H bytes each.
PreprocessBuffers allocPreprocessBuffers(int width, int height);
void              freePreprocessBuffers(PreprocessBuffers& b);

// Upload kernel weight constants (Gaussian + adaptive kernel).
// Safe to call multiple times — only uploads on first call.
void initKernelWeights();

// ─── Main entry point ────────────────────────────────────────────────────────
// Runs the full 6-kernel pipeline entirely on the GPU.
//
//   d_bgr    : [W*H*3] BGR image already on device
//   d_thresh : [W*H]   pre-allocated binary output on device
//   bufs     : pre-allocated intermediate buffers (must be >= W*H each)
//   stream   : CUDA stream — all kernels are enqueued asynchronously
//
// The function returns immediately without synchronising.
// Call cudaStreamSynchronize(stream) when you need d_thresh to be ready.
void preprocessDevice(const unsigned char* d_bgr,
                      unsigned char*       d_thresh,
                      int                  W,
                      int                  H,
                      PreprocessBuffers&   bufs,
                      cudaStream_t         stream = 0);
