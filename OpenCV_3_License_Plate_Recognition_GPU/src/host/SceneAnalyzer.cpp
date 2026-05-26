// SceneAnalyzer.cpp
// Stage 1: scene BGR → plate crops on GPU.

#include "SceneAnalyzer.h"
#include "Preprocess.cuh"
#include "CCL.cuh"
#include "ImageOps.cuh"
#include "CharGrouping.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// =============================================================================
std::vector<PossiblePlate> SceneAnalyzer::detectPlates(SceneBuffer&     sb,
                                                        PipelineContext& ctx)
{
    const int W = SCENE_W, H = SCENE_H;
    cudaStream_t stream = ctx.sceneStream;

    // ── Preprocess: BGR → binary threshold ───────────────────────────────────
    preprocessDevice(sb.d_scene_bgr, sb.d_scene_thresh, W, H, sb.scenePreproc, stream);

    // ── CCL + GPU filter → async D2H to pinned scene buffers ─────────────────
    cudaMemsetAsync(sb.d_num_filtered, 0, sizeof(int), stream);
    runCCLWithFilter(sb.d_scene_thresh, W, H,
                     sb.d_filtered, sb.d_num_filtered,
                     sb.h_filtered, sb.h_num_filtered,
                     stream);

    // Sync scene stream: we need h_filtered before CPU grouping.
    cudaStreamSynchronize(stream);

    // ── CPU: convert blobs → chars, geometric grouping ────────────────────────
    int numFiltered = *sb.h_num_filtered;
    auto chars      = blobsToChars(sb.h_filtered, numFiltered);
    auto groups     = groupChars(chars);

    // ── Batch WarpCrop: all plate groups → pool slots ─────────────────────────
    std::vector<PossiblePlate> plates;
    int slot = 0;
    std::vector<WarpParams>          warpParams;
    std::vector<unsigned char*>      dsts;

    for (auto& group : groups) {
        if (slot >= ctx.maxPlates) break;

        PossiblePlate plate = makePlate(group, slot, W, H);
        if (plate.poolSlot < 0) continue;   // rejected (dimensions out of bounds)

        warpParams.push_back({
            plate.rrLocationOfPlateInScene.center.x,
            plate.rrLocationOfPlateInScene.center.y,
            plate.rrLocationOfPlateInScene.angleDeg,
            plate.plateBgrW,
            plate.plateBgrH
        });
        dsts.push_back(ctx.plateBuffers[slot].d_plate_bgr);
        plates.push_back(plate);
        slot++;
    }

    if (!warpParams.empty())
        runBatchWarpCrop(sb.d_scene_bgr, W, H,
                         warpParams.data(), dsts.data(),
                         (int)warpParams.size(), stream);

    cudaStreamSynchronize(stream);

    std::cout << plates.size() << " plate candidate(s) found\n";
    return plates;
}

// =============================================================================
PossiblePlate SceneAnalyzer::makePlate(const std::vector<PossibleChar>& group,
                                        int slot, int srcW, int srcH)
{
    PossiblePlate plate;

    auto sorted = group;
    std::sort(sorted.begin(), sorted.end(), PossibleChar::sortCharsLeftToRight);

    const PossibleChar& first = sorted.front();
    const PossibleChar& last  = sorted.back();

    float cx = (first.intCenterX + last.intCenterX) * 0.5f;
    float cy = (first.intCenterY + last.intCenterY) * 0.5f;

    int plateW = (int)((last.boundingRect.x + last.boundingRect.width
                        - first.boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);
    double totalH = 0.0;
    for (const auto& c : sorted) totalH += c.boundingRect.height;
    int plateH = (int)((totalH / sorted.size()) * PLATE_HEIGHT_PADDING_FACTOR);

    // Reject if dimensions exceed pool allocation.
    if (plateW <= 0 || plateH <= 0 || plateW > MAX_PLATE_W || plateH > MAX_PLATE_H)
        return plate;   // poolSlot stays -1

    double opp      = last.intCenterY - first.intCenterY;
    double hyp      = distanceBetweenChars(first, last);
    double angleDeg = (hyp > 1e-6) ? std::asin(opp / hyp) * (180.0 / 3.14159265358979) : 0.0;

    plate.rrLocationOfPlateInScene = { {cx, cy}, {(float)plateW, (float)plateH}, (float)angleDeg };
    plate.poolSlot  = slot;
    plate.plateBgrW = plateW;
    plate.plateBgrH = plateH;
    return plate;
}
