// Types.h
// Plain C++ replacements for cv::Rect, cv::Point2f, cv::RotatedRect, cv::Size.
// No external dependencies.

#pragma once
#include <cmath>
#include <cstdint>

struct Rect2i {
    int x, y, width, height;
    int area() const { return width * height; }
};

struct Point2f {
    float x, y;
};

struct Size2f {
    float width, height;
};

// Encodes a rotated rectangle as center + (unrotated) dimensions + rotation angle.
// angleDeg follows the same sign convention as cv::RotatedRect: positive = clockwise.
struct RotatedRect {
    Point2f center;
    Size2f  size;
    float   angleDeg;
};
