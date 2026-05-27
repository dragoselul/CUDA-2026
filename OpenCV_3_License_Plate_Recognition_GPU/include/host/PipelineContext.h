// PipelineContext.h
// Owns all persistent GPU resources for the license-plate pipeline:
//   - Per-plate buffer pool (pre-allocated to avoid cudaMalloc in hot path)
//   - CCL workspaces (pre-allocated, one per slot — eliminates per-frame malloc)
//   - CUDA streams (1 scene stream + 1 transfer stream + maxPlates plate streams)
//   - Pinned host mirrors for fast async D2H transfers

#pragma once
#include "Types.h"
#include "CCL.cuh"
#include "KNN.cuh"
#include "Preprocess.cuh"
#include <vector>
#include <cuda_runtime.h>

// ─── Sizing constants ─────────────────────────────────────────────────────────
static constexpr int SCENE_W             = 1280;
static constexpr int SCENE_H             = 720;
static constexpr int MAX_PLATE_W         = 600;
static constexpr int MAX_PLATE_H         = 200;
static constexpr int MAX_PLATE_THRESH_W  = (int)(MAX_PLATE_W  * 1.6f + 1);  // 961
static constexpr int MAX_PLATE_THRESH_H  = (int)(MAX_PLATE_H  * 1.6f + 1);  // 321
static constexpr int MAX_CHARS_PER_PLATE = 20;
static constexpr int NUM_SCENE_SLOTS     = 4;   // ping-pong depth

// ─── Per-plate GPU / pinned buffer slot ──────────────────────────────────────
struct PlateBuffer {
    // Preprocessing intermediates (MAX_PLATE_W × MAX_PLATE_H each)
    PreprocessBuffers preproc;

    // WarpCrop output: BGR crop on device  [MAX_PLATE_W × MAX_PLATE_H × 3]
    unsigned char* d_plate_bgr   = nullptr;

    // Preprocess output (binary)            [MAX_PLATE_W × MAX_PLATE_H]
    unsigned char* d_thresh      = nullptr;

    // After resize ×1.6 (grayscale)        [MAX_PLATE_THRESH_W × MAX_PLATE_THRESH_H]
    unsigned char* d_thresh_big  = nullptr;

    // After Otsu threshold (binary)         [MAX_PLATE_THRESH_W × MAX_PLATE_THRESH_H]
    unsigned char* d_thresh_otsu = nullptr;

    // Pre-allocated CCL workspace (eliminates per-frame cudaMalloc)
    CCLWorkspace   plateWS       = {};

    // CCL filter output — device buffer + pinned host mirror
    FilteredBlob*  d_filtered        = nullptr;  // device [CCL_MAX_FILTERED]
    int*           d_num_filtered    = nullptr;  // device [1]
    FilteredBlob*  h_filtered        = nullptr;  // pinned [CCL_MAX_FILTERED]
    int*           h_num_filtered    = nullptr;  // pinned [1]

    // KNN inputs/outputs
    Rect2i*  d_rects      = nullptr;   // [MAX_CHARS_PER_PLATE] device
    float*   d_queries    = nullptr;   // [MAX_CHARS_PER_PLATE × KNN_FEATURES] device
    int32_t* d_knn_results= nullptr;   // [MAX_CHARS_PER_PLATE] device
    int32_t* h_labels     = nullptr;   // pinned [MAX_CHARS_PER_PLATE]

    void allocate();
    void free();
};

// ─── Scene-level buffers (NUM_SCENE_SLOTS sets, ping-pong) ───────────────────
struct SceneBuffer {
    // Scene image: device buffer, uploaded via cudaMemcpyAsync from loader
    unsigned char* d_scene_bgr    = nullptr;  // device [SCENE_W × SCENE_H × 3]

    // Device-only intermediate
    unsigned char* d_scene_thresh = nullptr;  // device [SCENE_W × SCENE_H]
    PreprocessBuffers scenePreproc = {};

    // Pre-allocated CCL workspace (eliminates per-frame cudaMalloc)
    CCLWorkspace   sceneWS         = {};

    // CCL filter output — device buffer + pinned host mirror
    FilteredBlob*  d_filtered      = nullptr;  // device [CCL_MAX_FILTERED]
    int*           d_num_filtered  = nullptr;  // device [1]
    FilteredBlob*  h_filtered      = nullptr;  // pinned [CCL_MAX_FILTERED]
    int*           h_num_filtered  = nullptr;  // pinned [1]

    void allocate();
    void free();
};

// ─── PipelineContext ──────────────────────────────────────────────────────────
struct PipelineContext {
    int maxPlates = 0;

    cudaStream_t              sceneStream    = nullptr;  // GPU compute
    cudaStream_t              transferStream = nullptr;  // H2D uploads
    cudaEvent_t               uploadDone[NUM_SCENE_SLOTS] = {};  // transfer → compute

    std::vector<cudaStream_t> plateStreams;
    std::vector<PlateBuffer>  plateBuffers;
    SceneBuffer               sceneBuffers[NUM_SCENE_SLOTS];

    static PipelineContext create(int maxPlates);
    void destroy();
};
