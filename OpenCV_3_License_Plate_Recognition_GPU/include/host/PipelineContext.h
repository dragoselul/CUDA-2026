// PipelineContext.h
// Owns all persistent GPU resources for the license-plate pipeline:
//   - KNN model (loaded once at startup)
//   - Per-plate buffer pool (pre-allocated to avoid cudaMalloc in hot path)
//   - CUDA streams (1 scene stream + maxPlates plate streams)
//   - Pinned host buffers for async D2H transfers

#pragma once
#include "Types.h"
#include "CCL.cuh"
#include "KNN.cuh"
#include "Preprocess.cuh"
#include <vector>
#include <cuda_runtime.h>

// ─── Sizing constants ─────────────────────────────────────────────────────────
static constexpr int SCENE_W             = 1280;  // fixed input resolution
static constexpr int SCENE_H             = 720;   // (enforce with normalize_images.py)
static constexpr int MAX_PLATE_W         = 600;
static constexpr int MAX_PLATE_H         = 200;
static constexpr int MAX_PLATE_THRESH_W  = (int)(MAX_PLATE_W  * 1.6f + 1);  // 961
static constexpr int MAX_PLATE_THRESH_H  = (int)(MAX_PLATE_H  * 1.6f + 1);  // 321
static constexpr int MAX_CHARS_PER_PLATE = 20;

// ─── Per-plate GPU/pinned buffer slot ─────────────────────────────────────────
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

    // CCL filter output on device           [CCL_MAX_FILTERED × FilteredBlob]
    FilteredBlob*  d_filtered        = nullptr;
    int*           d_num_filtered    = nullptr;

    // Pinned host mirrors for async D2H
    FilteredBlob*  h_filtered        = nullptr;
    int*           h_num_filtered    = nullptr;

    // KNN inputs/outputs (all pre-allocated, no per-call malloc)
    Rect2i*  d_rects      = nullptr;   // [MAX_CHARS_PER_PLATE] device rect buffer
    float*   d_queries    = nullptr;   // [MAX_CHARS_PER_PLATE × KNN_FEATURES]
    int32_t* d_knn_results= nullptr;   // [MAX_CHARS_PER_PLATE]
    int32_t* h_labels     = nullptr;   // pinned [MAX_CHARS_PER_PLATE]

    void allocate();
    void free();
};

// ─── Scene-level buffers (one set, reused per image) ─────────────────────────
struct SceneBuffer {
    FilteredBlob*     h_filtered     = nullptr;  // pinned [CCL_MAX_FILTERED]
    int*              h_num_filtered = nullptr;  // pinned [1]
    FilteredBlob*     d_filtered     = nullptr;  // device [CCL_MAX_FILTERED]
    int*              d_num_filtered = nullptr;  // device [1]

    unsigned char*    d_scene_bgr    = nullptr;  // device [SCENE_W × SCENE_H × 3]
    unsigned char*    d_scene_thresh = nullptr;  // device [SCENE_W × SCENE_H]
    PreprocessBuffers scenePreproc   = {};

    void allocate();
    void free();
};

// ─── PipelineContext ──────────────────────────────────────────────────────────
struct PipelineContext {
    int       maxPlates = 0;

    cudaStream_t              sceneStream    = nullptr;  // GPU compute
    cudaStream_t              transferStream = nullptr;  // H2D uploads only
    cudaEvent_t               uploadDone[2]  = {};       // handoff: transfer → compute

    std::vector<cudaStream_t> plateStreams;
    std::vector<PlateBuffer>  plateBuffers;
    SceneBuffer               sceneBuffers[2];           // ping-pong scene slots

    // Create a fully initialised context for up to `maxPlates` concurrent plates.
    static PipelineContext create(int maxPlates);
    void destroy();
};

