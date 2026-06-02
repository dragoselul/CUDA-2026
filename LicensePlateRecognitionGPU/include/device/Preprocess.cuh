#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct PreprocessBuffers {
    unsigned char* d_value;       // BGR → V (max channel)
    unsigned char* d_eroded;
    unsigned char* d_opened;      // dilate(erode(V)) = opening
    unsigned char* d_dilated;
    unsigned char* d_closed;      // erode(dilate(V)) = closing
    unsigned char* d_contrasted;
    unsigned char* d_blurred;
};

PreprocessBuffers allocPreprocessBuffers(int width, int height);
void              freePreprocessBuffers(PreprocessBuffers& b);

void initKernelWeights();

void preprocessDevice(const unsigned char* d_bgr,
                      unsigned char*       d_thresh,
                      int                  W,
                      int                  H,
                      PreprocessBuffers&   bufs,
                      cudaStream_t         stream = 0);
