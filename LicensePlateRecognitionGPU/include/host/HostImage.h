#pragma once
#include <cstddef>

struct HostImage {
    unsigned char* data;
    int width;
    int height;
    int channels;
    size_t step; // width * channels (usually)

    HostImage() : data(nullptr), width(0), height(0), channels(0), step(0) {}

    // Constructor to allocate memory
    HostImage(int w, int h, int c) : width(w), height(h), channels(c) {
        step = width * channels;
        data = new unsigned char[step * height];
    }

    HostImage(const HostImage&) = delete;
    HostImage& operator=(const HostImage&) = delete;

    HostImage(HostImage&& other) noexcept
        : data(other.data), width(other.width), height(other.height), channels(other.channels), step(other.step) {
        other.data = nullptr;
        other.width = 0;
        other.height = 0;
        other.channels = 0;
        other.step = 0;
    }

    HostImage& operator=(HostImage&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            step = other.step;

            other.data = nullptr;
            other.width = 0;
            other.height = 0;
            other.channels = 0;
            other.step = 0;
        }
        return *this;
    }

    // Access a pixel: image(x, y)
    unsigned char* ptr(int x, int y) {
        return data + (y * step) + (x * channels);
    }

    ~HostImage() { delete[] data; }
};