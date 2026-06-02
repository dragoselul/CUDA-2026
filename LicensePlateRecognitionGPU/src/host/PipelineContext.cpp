// PipelineContext.cpp
// Allocate, resize, and destroy all persistent GPU / pinned-host resources.

#include "PipelineContext.h"

// ─── PlateBuffer ─────────────────────────────────────────────────────────────
void PlateBuffer::allocate()
{
    preproc = allocPreprocessBuffers(MAX_PLATE_W, MAX_PLATE_H);

    cudaMalloc(&d_plate_bgr,    (size_t)MAX_PLATE_W        * MAX_PLATE_H        * 3);
    cudaMalloc(&d_thresh,       (size_t)MAX_PLATE_W        * MAX_PLATE_H);
    cudaMalloc(&d_thresh_big,   (size_t)MAX_PLATE_THRESH_W * MAX_PLATE_THRESH_H);
    cudaMalloc(&d_thresh_otsu,  (size_t)MAX_PLATE_THRESH_W * MAX_PLATE_THRESH_H);

    // Pre-allocated CCL workspace — sized for the post-resize plate image
    plateWS = allocWorkspace(MAX_PLATE_THRESH_W * MAX_PLATE_THRESH_H);

    // Device buffer + pinned host mirror for CCL filter output
    cudaMalloc    (&d_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMalloc    (&d_num_filtered, sizeof(int));
    cudaMallocHost(&h_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMallocHost(&h_num_filtered, sizeof(int));

    cudaMalloc(&d_rects,        MAX_CHARS_PER_PLATE * sizeof(Rect2i));
    cudaMalloc(&d_queries,      (size_t)MAX_CHARS_PER_PLATE * KNN_FEATURES * sizeof(float));
    cudaMalloc(&d_knn_results,  MAX_CHARS_PER_PLATE * sizeof(int32_t));
    cudaMallocHost(&h_labels,   MAX_CHARS_PER_PLATE * sizeof(int32_t));
}

void PlateBuffer::free()
{
    freePreprocessBuffers(preproc);
    cudaFree(d_plate_bgr);   cudaFree(d_thresh);
    cudaFree(d_thresh_big);  cudaFree(d_thresh_otsu);
    freeWorkspace(plateWS);
    cudaFree(d_filtered);    cudaFree(d_num_filtered);
    cudaFreeHost(h_filtered); cudaFreeHost(h_num_filtered);
    cudaFree(d_rects);       cudaFree(d_queries);  cudaFree(d_knn_results);
    cudaFreeHost(h_labels);
    *this = {};
}

// ─── SceneBuffer ─────────────────────────────────────────────────────────────
void SceneBuffer::allocate()
{
    // Device buffer: loader uploads via cudaMemcpyAsync on transferStream.
    cudaMalloc(&d_scene_bgr,    (size_t)SCENE_W * SCENE_H * 3);
    cudaMalloc(&d_scene_thresh, (size_t)SCENE_W * SCENE_H);
    scenePreproc = allocPreprocessBuffers(SCENE_W, SCENE_H);

    // Pre-allocated CCL workspace — sized for the full scene image
    sceneWS = allocWorkspace(SCENE_W * SCENE_H);

    // Device buffer + pinned host mirror for CCL filter output
    cudaMalloc    (&d_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMalloc    (&d_num_filtered, sizeof(int));
    cudaMallocHost(&h_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMallocHost(&h_num_filtered, sizeof(int));
}

void SceneBuffer::free()
{
    cudaFree(d_scene_bgr);
    cudaFree(d_scene_thresh);
    freePreprocessBuffers(scenePreproc);
    freeWorkspace(sceneWS);
    cudaFree(d_filtered);    cudaFree(d_num_filtered);
    cudaFreeHost(h_filtered); cudaFreeHost(h_num_filtered);
    *this = {};
}

// ─── PipelineContext ──────────────────────────────────────────────────────────
PipelineContext PipelineContext::create(int maxPlates)
{
    PipelineContext ctx;
    ctx.maxPlates = maxPlates;

    cudaStreamCreate(&ctx.sceneStream);
    cudaStreamCreate(&ctx.transferStream);
    for (auto& e : ctx.uploadDone) cudaEventCreate(&e);

    ctx.plateStreams.resize(maxPlates);
    for (auto& s : ctx.plateStreams) cudaStreamCreate(&s);

    ctx.plateBuffers.resize(maxPlates);
    for (auto& pb : ctx.plateBuffers) pb.allocate();

    for (auto& sb : ctx.sceneBuffers) sb.allocate();

    return ctx;
}

void PipelineContext::destroy()
{
    cudaStreamDestroy(sceneStream);
    cudaStreamDestroy(transferStream);
    for (auto e : uploadDone)  cudaEventDestroy(e);
    for (auto s : plateStreams) cudaStreamDestroy(s);
    for (auto& pb : plateBuffers) pb.free();
    for (auto& sb : sceneBuffers) sb.free();
}
