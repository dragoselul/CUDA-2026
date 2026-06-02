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
    static std::vector<PossibleChar> bestCharGroup(const FilteredBlob* blobs,
                                                    int count);

    static std::string assembleString(const int32_t* labels, int count);
};
