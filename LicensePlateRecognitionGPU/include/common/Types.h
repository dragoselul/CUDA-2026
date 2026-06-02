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

struct RotatedRect {
    Point2f center;
    Size2f  size;
    float   angleDeg;
};
