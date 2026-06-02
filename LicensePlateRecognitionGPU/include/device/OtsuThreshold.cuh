#pragma once
#include <cuda_runtime.h>

__global__ void histogramKernel(
    const unsigned char* d_input,
    int                  numPixels,
    int*                 d_hist);

__global__ void applyThresholdKernel(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  numPixels,
    unsigned char        thresh);

unsigned char computeOtsuThreshold(const int* h_hist, int numPixels);

void runOtsuThreshold(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  width,
    int                  height,
    cudaStream_t         stream = 0);
