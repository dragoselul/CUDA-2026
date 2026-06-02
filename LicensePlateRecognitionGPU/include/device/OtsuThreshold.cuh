// OtsuThreshold.cuh
// GPU histogram + CPU Otsu threshold computation + GPU threshold apply.
// Replaces cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU).

#pragma once
#include <cuda_runtime.h>

// Compute a 256-bin histogram using shared-memory accumulation.
// d_hist must be zeroed by the caller (cudaMemset(d_hist, 0, 256 * sizeof(int))).
__global__ void histogramKernel(
    const unsigned char* d_input,
    int                  numPixels,
    int*                 d_hist);

// Apply binary threshold (THRESH_BINARY): pixel >= thresh -> 255, else -> 0.
__global__ void applyThresholdKernel(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  numPixels,
    unsigned char        thresh);

// CPU-side Otsu computation from a 256-bin histogram.
// Returns the optimal threshold value.
unsigned char computeOtsuThreshold(const int* h_hist, int numPixels);

// Host launcher: runs the full pipeline on an image already on the device.
// d_input  : source grayscale image on device (numPixels bytes)
// d_output : destination image on device (numPixels bytes, must be pre-allocated)
void runOtsuThreshold(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  width,
    int                  height,
    cudaStream_t         stream = 0);
