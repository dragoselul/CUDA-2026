// PlateRecognizer.h
// Stage 2 of the pipeline: plate BGR crops on GPU → recognised character strings.
//
// Each plate runs on its own CUDA stream (ctx.plateStreams[plate.poolSlot]):
//   preprocess → resize ×1.6 → OtsuThreshold → CCL+filter (async D2H)
// After all plates finish (cudaDeviceSynchronize), CPU groups the surviving
// blobs, then a second GPU wave does charROIResize + KNN (async D2H).
// A final sync yields the recognised labels.

#pragma once
#include "PipelineContext.h"
#include "PossiblePlate.h"
#include "PossibleChar.h"
#include <vector>
#include <string>

class PlateRecognizer {
public:
    void recognizePlates(std::vector<PossiblePlate>& plates,
                         PipelineContext&             ctx);

private:
    // CPU char grouping for one plate's filtered blobs.
    static std::vector<PossibleChar> bestCharGroup(const FilteredBlob* blobs,
                                                    int count);

    // Assemble recognised string from pinned label array.
    static std::string assembleString(const int32_t* labels, int count);
};
