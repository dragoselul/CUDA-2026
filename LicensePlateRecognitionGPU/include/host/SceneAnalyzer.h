#pragma once
#include "PipelineContext.h"
#include "PossibleChar.h"
#include "PossiblePlate.h"
#include <vector>

class SceneAnalyzer {
public:
    std::vector<PossiblePlate> detectPlates(SceneBuffer& sb,
                                            PipelineContext& ctx);

private:
    static constexpr double PLATE_WIDTH_PADDING_FACTOR  = 1.3;
    static constexpr double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

    static PossiblePlate makePlate(const std::vector<PossibleChar>& group,
                                   int slot, int srcW, int srcH);
};
