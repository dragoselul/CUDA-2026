// SceneAnalyzer.h
// Stage 1 of the pipeline: full scene BGR → plate candidate crops on the GPU.
// All GPU work runs on PipelineContext::sceneStream.
// The caller is responsible for uploading scene BGR into sb.d_scene_bgr before
// calling detectPlates (via PipelineContext::transferStream + uploadDone event).

#pragma once
#include "PipelineContext.h"
#include "PossibleChar.h"
#include "PossiblePlate.h"
#include <vector>

class SceneAnalyzer {
public:
    // Preprocess, CCL+filter on GPU, group chars on CPU, then batch-WarpCrop
    // each plate group into ctx.plateBuffers[i].d_plate_bgr.
    // sb.d_scene_bgr must already be uploaded before this call.
    // Returns PossiblePlate[] with poolSlot and dimension fields set.
    std::vector<PossiblePlate> detectPlates(SceneBuffer& sb,
                                            PipelineContext& ctx);

private:
    static constexpr double PLATE_WIDTH_PADDING_FACTOR  = 1.3;
    static constexpr double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

    // Build PossiblePlate from a matched char group; fills poolSlot + dimensions.
    static PossiblePlate makePlate(const std::vector<PossibleChar>& group,
                                   int slot, int srcW, int srcH);
};
