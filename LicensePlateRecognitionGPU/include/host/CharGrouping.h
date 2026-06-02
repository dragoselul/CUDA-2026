// CharGrouping.h
// Shared geometric helpers used by both SceneAnalyzer and PlateRecognizer.
// All functions are pure (no global state) and CPU-only.

#pragma once
#include "PossibleChar.h"
#include "CCL.cuh"
#include <algorithm>
#include <vector>
#include <cmath>

// ─── Filter thresholds (character geometry) ───────────────────────────────────
static constexpr int    MIN_NUMBER_OF_MATCHING_CHARS = 3;
static constexpr double MIN_DIAG_MULTIPLE_AWAY       = 0.3;
static constexpr double MAX_DIAG_MULTIPLE_AWAY       = 5.0;
static constexpr double MAX_CHANGE_IN_AREA           = 0.5;
static constexpr double MAX_CHANGE_IN_WIDTH          = 0.8;
static constexpr double MAX_CHANGE_IN_HEIGHT         = 0.2;
static constexpr double MAX_ANGLE_BETWEEN_CHARS      = 12.0;

// Convert a GPU-filtered blob to a PossibleChar (CPU).
inline PossibleChar blobToChar(const FilteredBlob& b)
{
    return PossibleChar({ b.x, b.y, b.width, b.height }, b.compactId);
}

// Convert the first `count` blobs to a PossibleChar vector.
inline std::vector<PossibleChar> blobsToChars(const FilteredBlob* blobs, int count)
{
    std::vector<PossibleChar> out;
    out.reserve(count);
    for (int i = 0; i < count; i++) out.push_back(blobToChar(blobs[i]));
    return out;
}

inline double distanceBetweenChars(const PossibleChar& a, const PossibleChar& b)
{
    double dx = a.intCenterX - b.intCenterX, dy = a.intCenterY - b.intCenterY;
    return std::sqrt(dx*dx + dy*dy);
}

inline double angleBetweenChars(const PossibleChar& a, const PossibleChar& b)
{
    double adj = std::abs(a.intCenterX - b.intCenterX);
    double opp = std::abs(a.intCenterY - b.intCenterY);
    if (adj < 1e-6) return 90.0;
    return std::atan(opp / adj) * (180.0 / 3.14159265358979);
}

inline std::vector<PossibleChar> findMatchingChars(
    const PossibleChar& pc, const std::vector<PossibleChar>& all)
{
    std::vector<PossibleChar> out;
    for (const auto& c : all) {
        if (c == pc) continue;
        double dist  = distanceBetweenChars(pc, c);
        double angle = angleBetweenChars(pc, c);
        double areaC = std::abs(c.boundingRect.area()  - pc.boundingRect.area())
                     / (double)pc.boundingRect.area();
        double widC  = std::abs(c.boundingRect.width  - pc.boundingRect.width)
                     / (double)pc.boundingRect.width;
        double htC   = std::abs(c.boundingRect.height - pc.boundingRect.height)
                     / (double)pc.boundingRect.height;
        if (dist  < pc.dblDiagonalSize * MAX_DIAG_MULTIPLE_AWAY
         && angle < MAX_ANGLE_BETWEEN_CHARS
         && areaC < MAX_CHANGE_IN_AREA
         && widC  < MAX_CHANGE_IN_WIDTH
         && htC   < MAX_CHANGE_IN_HEIGHT)
            out.push_back(c);
    }
    return out;
}

inline std::vector<std::vector<PossibleChar>> groupChars(
    const std::vector<PossibleChar>& chars)
{
    std::vector<std::vector<PossibleChar>> result;
    for (const auto& pc : chars) {
        auto matching = findMatchingChars(pc, chars);
        matching.push_back(pc);
        if ((int)matching.size() < MIN_NUMBER_OF_MATCHING_CHARS) continue;
        result.push_back(matching);

        std::vector<PossibleChar> remaining;
        for (const auto& c : chars)
            if (std::find(matching.begin(), matching.end(), c) == matching.end())
                remaining.push_back(c);

        for (auto& r : groupChars(remaining)) result.push_back(r);
        break;
    }
    return result;
}

inline std::vector<PossibleChar> removeInnerOverlapping(std::vector<PossibleChar> chars)
{
    std::vector<PossibleChar> result = chars;
    for (const auto& a : chars)
        for (const auto& b : chars) {
            if (a == b) continue;
            if (distanceBetweenChars(a, b) < a.dblDiagonalSize * MIN_DIAG_MULTIPLE_AWAY) {
                const PossibleChar& drop = (a.boundingRect.area() < b.boundingRect.area()) ? a : b;
                auto it = std::find(result.begin(), result.end(), drop);
                if (it != result.end()) result.erase(it);
            }
        }
    return result;
}
