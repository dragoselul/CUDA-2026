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

bool loadKNNModel(const char* binPath, KNNModel& model);
void freeKNNModel(KNNModel& model);
void runKNNDevice(const KNNModel& model,
                  const float*    d_queries,
                  int             numChars,
                  int32_t*        d_results,
                  int32_t*        h_labels,
                  cudaStream_t    stream = 0);
