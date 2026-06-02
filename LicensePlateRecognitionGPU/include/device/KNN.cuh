// KNN.cuh
// GPU brute-force k=1 nearest neighbor for character recognition.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

static constexpr int KNN_CHAR_W   = 20;
static constexpr int KNN_CHAR_H   = 30;
static constexpr int KNN_FEATURES = KNN_CHAR_W * KNN_CHAR_H;   // 600

struct KNNModel {
    float*   d_training    = nullptr;   // [numSamples × KNN_FEATURES] float32, device
    int32_t* d_trainLabels = nullptr;   // [numSamples] ASCII int32, device
    int      numSamples    = 0;
};

// Load binary knn_data.bin; allocates GPU memory. Returns false on failure.
bool loadKNNModel(const char* binPath, KNNModel& model);
void freeKNNModel(KNNModel& model);

// Classify numChars query images (d_queries already on device) using k=1 NN.
// d_results   : pre-allocated device buffer [numChars × int32_t]
// h_labels    : pre-allocated pinned host output [numChars] — filled via async D2H
//
// All work is enqueued on `stream`; no internal synchronisation.
// Caller must cudaStreamSynchronize(stream) before reading h_labels.
void runKNNDevice(const KNNModel& model,
                  const float*    d_queries,
                  int             numChars,
                  int32_t*        d_results,
                  int32_t*        h_labels,
                  cudaStream_t    stream = 0);
