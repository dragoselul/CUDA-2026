// ImageIO.h
// Minimal PPM (P6 binary) image loader. No external dependencies.

#pragma once
#include "HostImage.h"
#include <string>

// Load a binary PPM (P6) file.
// Returns a BGR HostImage (R and B bytes are swapped from the file's RGB layout).
// On failure (file not found, wrong format), returns a HostImage with data == nullptr.
HostImage loadPPM(const std::string& path);
