// PossibleChar.h
// No external dependencies. Contour replaced by bounding rect + component ID.

#ifndef POSSIBLE_CHAR_H
#define POSSIBLE_CHAR_H

#include "Types.h"
#include <cmath>

class PossibleChar {
public:
    Rect2i  boundingRect;
    int     intCenterX;
    int     intCenterY;
    double  dblDiagonalSize;
    double  dblAspectRatio;
    int     componentId;   // unique CCL component ID; used for equality

    PossibleChar() = default;

    // Construct from a CCL bounding box and compact component ID.
    PossibleChar(const Rect2i& rect, int compId)
        : boundingRect(rect),
          intCenterX(rect.x + rect.width  / 2),
          intCenterY(rect.y + rect.height / 2),
          dblDiagonalSize(std::sqrt((double)rect.width  * rect.width
                                  + (double)rect.height * rect.height)),
          dblAspectRatio((rect.height > 0) ? (double)rect.width / (double)rect.height : 0.0),
          componentId(compId) {}

    bool operator==(const PossibleChar& o) const { return componentId == o.componentId; }
    bool operator!=(const PossibleChar& o) const { return componentId != o.componentId; }

    static bool sortCharsLeftToRight(const PossibleChar& a, const PossibleChar& b) {
        return a.intCenterX < b.intCenterX;
    }
};

#endif  // POSSIBLE_CHAR_H
