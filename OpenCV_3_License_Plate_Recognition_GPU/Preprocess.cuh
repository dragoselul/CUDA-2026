
#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <vector>
#include "HostImage.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
#define BLOCK_W 16
#define BLOCK_H 16
#define ADAPTIVE_C 2.0f

// function prototypes ////////////////////////////////////////////////////////////////////////////

struct GPUImageBuffers {
    unsigned char* d_value;         // after BGR → V
    unsigned char* d_eroded;        // after erode(value)
    unsigned char* d_opened;        // after dilate(eroded)    = opening
    unsigned char* d_dilated;       // after dilate(value)
    unsigned char* d_closed;        // after erode(dilated)    = closing
    unsigned char* d_contrasted;    // after top hat / black hat
    unsigned char* d_blurred;       // after gaussian blur
    // d_output is kept separate per-image (see preprocess())
};

GPUImageBuffers allocateGPUBuffers(int width, int height);
void freeGPUBuffers(GPUImageBuffers& b);


void initKernelWeights();

void preprocess(const std::vector<HostImage> &imgOriginalBatch,
                     std::vector<HostImage> &imgGrayscaleBatch,
                     std::vector<HostImage> &imgThreshBatch,
                     int maxConcurrentStreams = 0);

#endif //PREPROCESS_CUH

