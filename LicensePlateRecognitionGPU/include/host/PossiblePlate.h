#pragma once
#include "Types.h"
#include <string>

class PossiblePlate {
public:
    RotatedRect rrLocationOfPlateInScene;
    std::string strChars;

    int poolSlot  = -1;
    int plateBgrW = 0;
    int plateBgrH = 0;

    static bool sortDescendingByNumberOfChars(const PossiblePlate& a,
                                              const PossiblePlate& b)
    { return a.strChars.length() > b.strChars.length(); }
};
