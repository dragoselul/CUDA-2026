// ImageIO.cpp
// Binary PPM (P6) loader — no external dependencies.

#include "ImageIO.h"
#include <fstream>
#include <sstream>
#include <iostream>

// Skip whitespace and '#' comment lines in the PPM header.
static void skipWhitespaceAndComments(std::istream& in) {
    while (in.good()) {
        char c = static_cast<char>(in.peek());
        if (c == '#') {
            std::string line;
            std::getline(in, line);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            in.get();
        } else {
            break;
        }
    }
}

HostImage loadPPM(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "loadPPM: cannot open " << path << "\n";
        return HostImage();
    }

    // Magic number
    std::string ppmFormat;
    f >> ppmFormat;
    if (ppmFormat != "P6") {
        std::cerr << "loadPPM: not a binary PPM (P6) file: " << path << "\n";
        return HostImage();
    }

    skipWhitespaceAndComments(f);
    int width = 0, height = 0, maxval = 0;
    f >> width;
    skipWhitespaceAndComments(f);
    f >> height;
    skipWhitespaceAndComments(f);
    f >> maxval;

    if (width <= 0 || height <= 0 || maxval != 255) {
        std::cerr << "loadPPM: unsupported PPM header in " << path
                  << " (w=" << width << " h=" << height << " maxval=" << maxval << ")\n";
        return HostImage();
    }

    // The single whitespace character after maxval before the pixel data
    char ws;
    f.get(ws);

    HostImage img(width, height, 3);
    if (!img.data) {
        std::cerr << "loadPPM: allocation failed\n";
        return HostImage();
    }

    const size_t totalBytes = static_cast<size_t>(width) * height * 3;
    f.read(reinterpret_cast<char*>(img.data), static_cast<std::streamsize>(totalBytes));
    if (!f) {
        std::cerr << "loadPPM: short read in " << path << "\n";
        return HostImage();
    }

    // PPM stores RGB; swap R and B to produce BGR expected by extractValueKernel.
    for (size_t i = 0; i < totalBytes; i += 3) {
        std::swap(img.data[i + 0], img.data[i + 2]);   // R <-> B
    }

    return img;
}
