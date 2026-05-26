// PipelineContext.cpp
// Allocate, resize, and destroy all persistent GPU/pinned-host resources.

#include "PipelineContext.h"

void PlateBuffer::allocate()
{
    preproc = allocPreprocessBuffers(MAX_PLATE_W, MAX_PLATE_H);

    cudaMalloc(&d_plate_bgr,    (size_t)MAX_PLATE_W        * MAX_PLATE_H        * 3);
    cudaMalloc(&d_thresh,       (size_t)MAX_PLATE_W        * MAX_PLATE_H);
    cudaMalloc(&d_thresh_big,   (size_t)MAX_PLATE_THRESH_W * MAX_PLATE_THRESH_H);
    cudaMalloc(&d_thresh_otsu,  (size_t)MAX_PLATE_THRESH_W * MAX_PLATE_THRESH_H);
    cudaMalloc(&d_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMalloc(&d_num_filtered, sizeof(int));

    cudaMallocHost(&h_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMallocHost(&h_num_filtered, sizeof(int));
    cudaMallocHost(&h_labels,       MAX_CHARS_PER_PLATE * sizeof(int32_t));

    cudaMalloc(&d_rects,        MAX_CHARS_PER_PLATE * sizeof(Rect2i));
    cudaMalloc(&d_queries,      (size_t)MAX_CHARS_PER_PLATE * KNN_FEATURES * sizeof(float));
    cudaMalloc(&d_knn_results,  MAX_CHARS_PER_PLATE * sizeof(int32_t));
}

void PlateBuffer::free()
{
    freePreprocessBuffers(preproc);
    cudaFree(d_plate_bgr);    cudaFree(d_thresh);
    cudaFree(d_thresh_big);   cudaFree(d_thresh_otsu);
    cudaFree(d_filtered);     cudaFree(d_num_filtered);
    cudaFreeHost(h_filtered); cudaFreeHost(h_num_filtered); cudaFreeHost(h_labels);
    cudaFree(d_rects);        cudaFree(d_queries); cudaFree(d_knn_results);
    *this = {};
}

void SceneBuffer::allocate()
{
    cudaMallocHost(&h_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMallocHost(&h_num_filtered, sizeof(int));
    cudaMalloc(&d_filtered,     CCL_MAX_FILTERED * sizeof(FilteredBlob));
    cudaMalloc(&d_num_filtered, sizeof(int));
    cudaMalloc(&d_scene_bgr,    (size_t)SCENE_W * SCENE_H * 3);
    cudaMalloc(&d_scene_thresh, (size_t)SCENE_W * SCENE_H);
    scenePreproc = allocPreprocessBuffers(SCENE_W, SCENE_H);
}

void SceneBuffer::free()
{
    cudaFreeHost(h_filtered); cudaFreeHost(h_num_filtered);
    cudaFree(d_filtered);     cudaFree(d_num_filtered);
    cudaFree(d_scene_bgr);    cudaFree(d_scene_thresh);
    freePreprocessBuffers(scenePreproc);
    *this = {};
}

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

    ctx.sceneBuffers[0].allocate();
    ctx.sceneBuffers[1].allocate();
    return ctx;
}

void PipelineContext::destroy()
{
    cudaStreamDestroy(sceneStream);
    cudaStreamDestroy(transferStream);
    for (auto e : uploadDone) cudaEventDestroy(e);
    for (auto s : plateStreams) cudaStreamDestroy(s);
    for (auto& pb : plateBuffers) pb.free();
    sceneBuffers[0].free();
    sceneBuffers[1].free();
}
