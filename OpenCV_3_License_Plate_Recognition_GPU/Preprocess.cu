// Preprocess.cu

#include "Preprocess.cuh"

#include <algorithm>
#include <vector>

#define BLOCK_W 16
#define BLOCK_H 16

__device__ int clampDevice(int val, int lo, int hi) {
    return max(lo, min(val, hi));
}

__device__ uint8_t getPixelDevice(const unsigned char* input,
                                   int x, int y, int width, int height) {
    return input[clampDevice(y, 0, height-1) * width
               + clampDevice(x, 0, width-1)];
}

__constant__ float c_gaussKernel[25];

// 15×15 adaptive threshold kernel — matches OpenCV ADAPTIVE_THRESH_GAUSSIAN_C
// with blockSize=15
__constant__ float c_adaptiveKernel[225];


void initKernelWeights() {

    // ── 5×5 Gaussian blur weights ─────────────────────────────────────────────
    {
        const int   ksize  = 5;
        const int   radius = 2;
        const float sigma  = 1.0f;
        float host[25];
        float sum = 0.f;
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                float dy = ky - radius;
                float dx = kx - radius;
                float w  = expf(-(dx*dx + dy*dy) / (2.f * sigma * sigma));
                host[ky * ksize + kx] = w;
                sum += w;
            }
        }
        for (int i = 0; i < 25; i++) host[i] /= sum;
        cudaMemcpyToSymbol(c_gaussKernel, host, sizeof(host));
    }

    // ── 15×15 adaptive threshold weights ─────────────────────────────────────
    {
        const int   ksize  = 15;
        const int   radius = 7;
        const float sigma  = (ksize / 2.0f - 1) * 0.3f + 0.8f;
        float host[225];
        float sum = 0.f;
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                float dy = ky - radius;
                float dx = kx - radius;
                float w  = expf(-(dx*dx + dy*dy) / (2.f * sigma * sigma));
                host[ky * ksize + kx] = w;
                sum += w;
            }
        }
        for (int i = 0; i < 225; i++) host[i] /= sum;
        cudaMemcpyToSymbol(c_adaptiveKernel, host, sizeof(host));
    }
}

__global__ void extractValueKernel(
    const uchar4* inputBGRA,
    unsigned char* outputV,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 pixel = inputBGRA[y * width + x];
    // V = max(R, G, B). OpenCV BGR maps to x=B, y=G, z=R.
    outputV[y * width + x] = max(pixel.x, max(pixel.y, pixel.z));
}

// ─── Kernel 2: Erosion (shared memory tiling) ────────────────────────────────

__global__ void erodeKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height, int kernelSize)
{
    if (kernelSize % 2 == 0) kernelSize++;
    const int radius = kernelSize / 2;
    const int tileW  = BLOCK_W + 2 * radius;
    const int tileH  = BLOCK_H + 2 * radius;

    extern __shared__ uint8_t s_tile[];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load tile + halo into shared memory
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampDevice(blockIdx.x * blockDim.x + tx - radius, 0, width  - 1);
            int gy = clampDevice(blockIdx.y * blockDim.y + ty - radius, 0, height - 1);
            s_tile[ty * tileW + tx] = input[gy * width + gx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    uint8_t minVal = 255;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            minVal = min(minVal,
                s_tile[(threadIdx.y + radius + dy) * tileW
                      + (threadIdx.x + radius + dx)]);

    output[y * width + x] = minVal;
}

// ─── Kernel 3: Dilation (shared memory tiling) ───────────────────────────────

__global__ void dilateKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height, int kernelSize)
{
    if (kernelSize % 2 == 0) kernelSize++;
    const int radius = kernelSize / 2;
    const int tileW  = BLOCK_W + 2 * radius;
    const int tileH  = BLOCK_H + 2 * radius;

    extern __shared__ uint8_t s_tile[];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampDevice(blockIdx.x * blockDim.x + tx - radius, 0, width  - 1);
            int gy = clampDevice(blockIdx.y * blockDim.y + ty - radius, 0, height - 1);
            s_tile[ty * tileW + tx] = input[gy * width + gx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    uint8_t maxVal = 0;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            maxVal = max(maxVal,
                s_tile[(threadIdx.y + radius + dy) * tileW
                      + (threadIdx.x + radius + dx)]);

    output[y * width + x] = maxVal;
}

// ─── Kernel 4: Top Hat + Black Hat ───────────────────────────────────────────
// Pointwise — no shared memory needed

__global__ void maximizeContrastKernel(
    const unsigned char* d_value,       // original grayscale
    const unsigned char* d_opened,      // result of erode→dilate
    const unsigned char* d_closed,      // result of dilate→erode
          unsigned char* d_contrasted,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i = y * width + x;
    int16_t original = d_value[i];
    int16_t topHat   = original        - (int16_t)d_opened[i];
    int16_t blackHat = (int16_t)d_closed[i] - original;
    int16_t result   = original + topHat - blackHat;

    d_contrasted[i] = (unsigned char)clampDevice((int)result, 0, 255);
}

// KERNEL 5 — Gaussian Blur (5×5)

__global__ void gaussianBlurKernel(
    const unsigned char* input,
          unsigned char* output,
    int width, int height)
{
    const int radius = 2;
    const int tileW  = BLOCK_W + 2 * radius;    // 20
    const int tileH  = BLOCK_H + 2 * radius;    // 20

    extern __shared__ uint8_t s_tile[];          // 400 bytes

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampDevice(blockIdx.x * blockDim.x + tx - radius, 0, width  - 1);
            int gy = clampDevice(blockIdx.y * blockDim.y + ty - radius, 0, height - 1);
            s_tile[ty * tileW + tx] = input[gy * width + gx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int kIdx = (dy + radius) * 5 + (dx + radius);
            acc += c_gaussKernel[kIdx]
                 * (float)s_tile[(threadIdx.y + radius + dy) * tileW
                                + (threadIdx.x + radius + dx)];
        }
    }

    output[y * width + x] = (unsigned char)clampDevice((int)acc, 0, 255);
}


// KERNEL 6 — Adaptive Gaussian Threshold

__global__ void adaptiveThresholdKernel(
    const unsigned char* input,
          unsigned char* output,
    int width, int height, float C)
{
    const int radius = 7;
    const int ksize  = 15;
    const int tileW  = BLOCK_W + 2 * radius;    // 30
    const int tileH  = BLOCK_H + 2 * radius;    // 30

    extern __shared__ uint8_t s_tile[];          // 900 bytes

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampDevice(blockIdx.x * blockDim.x + tx - radius, 0, width  - 1);
            int gy = clampDevice(blockIdx.y * blockDim.y + ty - radius, 0, height - 1);
            s_tile[ty * tileW + tx] = input[gy * width + gx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    // Compute local weighted mean
    float mean = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int kIdx = (dy + radius) * ksize + (dx + radius);
            mean += c_adaptiveKernel[kIdx]
                  * (float)s_tile[(threadIdx.y + radius + dy) * tileW
                                 + (threadIdx.x + radius + dx)];
        }
    }

    // Pixel value is at the centre of the tile
    float pixel = (float)s_tile[(threadIdx.y + radius) * tileW + (threadIdx.x + radius)];

    // THRESH_BINARY_INV: 255 where pixel is below local threshold
    output[y * width + x] = (pixel < (mean - C)) ? 255 : 0;
}

// ─── Host: persistent GPU buffer struct ──────────────────────────────────────
// Allocate once, reuse across calls — this is the key to staying on the GPU

GPUImageBuffers allocateGPUBuffers(int width, int height) {
    GPUImageBuffers b;
    size_t sz = width * height * sizeof(unsigned char);
    cudaMalloc(&b.d_value,      sz);
    cudaMalloc(&b.d_eroded,     sz);
    cudaMalloc(&b.d_opened,     sz);
    cudaMalloc(&b.d_dilated,    sz);
    cudaMalloc(&b.d_closed,     sz);
    cudaMalloc(&b.d_contrasted, sz);
    cudaMalloc(&b.d_blurred,    sz);
    return b;
}

void freeGPUBuffers(GPUImageBuffers& b) {
    cudaFree(b.d_value);
    cudaFree(b.d_eroded);
    cudaFree(b.d_opened);
    cudaFree(b.d_dilated);
    cudaFree(b.d_closed);
    cudaFree(b.d_contrasted);
    cudaFree(b.d_blurred);
}

// ─── Host: pipeline launcher ──────────────────────────────────────────────────

void preprocess(const std::vector<HostImage> &imgOriginalBatch,
                     std::vector<HostImage> &imgGrayscaleBatch,
                     std::vector<HostImage> &imgThreshBatch,
                     int maxConcurrentStreams) {

    static bool initialized = false;
    if (!initialized) {
        initKernelWeights();
        initialized = true;
    }

    const int total = static_cast<int>(imgOriginalBatch.size());
    imgGrayscaleBatch.clear();
    imgThreshBatch.clear();
    if (total == 0) return;

    imgGrayscaleBatch.reserve(total);
    imgThreshBatch.reserve(total);
    for (const HostImage& img : imgOriginalBatch) {
        imgGrayscaleBatch.emplace_back(img.width, img.height, 1);
        imgThreshBatch.emplace_back(img.width, img.height, 1);
    }

    int streamsPerWave = (maxConcurrentStreams > 0) ? maxConcurrentStreams : 8;
    streamsPerWave = std::max(1, std::min(streamsPerWave, total));

    const int kernelSize = 3;
    dim3 block(BLOCK_W, BLOCK_H);

    const int morphRadius = 1;   // 3x3
    const int blurRadius = 2;    // 5x5
    const int adaptRadius = 7;   // 15x15

    const size_t morphShared = (BLOCK_W + 2 * morphRadius) * (BLOCK_H + 2 * morphRadius) * sizeof(uint8_t);
    const size_t blurShared = (BLOCK_W + 2 * blurRadius) * (BLOCK_H + 2 * blurRadius) * sizeof(uint8_t);
    const size_t adaptShared = (BLOCK_W + 2 * adaptRadius) * (BLOCK_H + 2 * adaptRadius) * sizeof(uint8_t);

    for (int waveStart = 0; waveStart < total; waveStart += streamsPerWave) {
        int waveEnd = std::min(waveStart + streamsPerWave, total);
        int waveCount = waveEnd - waveStart;

        std::vector<cudaStream_t> streams(waveCount);
        std::vector<GPUImageBuffers> bufs(waveCount);
        std::vector<uchar4*> d_input(waveCount, nullptr);
        std::vector<unsigned char*> d_output(waveCount, nullptr);
        std::vector<std::vector<uchar4>> inputBGRA(waveCount);

        for (int slot = 0; slot < waveCount; ++slot) {
            const int i = waveStart + slot;
            const HostImage& srcImg = imgOriginalBatch[i];

            const int width = srcImg.width;
            const int height = srcImg.height;
            const size_t bgraSize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4);
            const size_t graySize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(unsigned char);

            cudaStreamCreate(&streams[slot]);
            bufs[slot] = allocateGPUBuffers(width, height);
            cudaMalloc(&d_input[slot], bgraSize);
            cudaMalloc(&d_output[slot], graySize);

            // Pack BGR (3 bytes) into BGRA (4 bytes) for aligned global reads.
            inputBGRA[slot].resize(static_cast<size_t>(width) * static_cast<size_t>(height));
            for (int y = 0; y < height; ++y) {
                const unsigned char* srcRow = srcImg.data + static_cast<size_t>(y) * srcImg.step;
                for (int x = 0; x < width; ++x) {
                    const unsigned char* p = srcRow + (x * srcImg.channels);
                    inputBGRA[slot][static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                        make_uchar4(p[0], p[1], p[2], 0);
                }
            }

            cudaMemcpyAsync(d_input[slot], inputBGRA[slot].data(), bgraSize, cudaMemcpyHostToDevice, streams[slot]);

            dim3 grid((width + BLOCK_W - 1) / BLOCK_W,
                      (height + BLOCK_H - 1) / BLOCK_H);

            extractValueKernel<<<grid, block, 0, streams[slot]>>>(
                d_input[slot], bufs[slot].d_value, width, height);

            // opening = dilate(erode(value))
            erodeKernel<<<grid, block, morphShared, streams[slot]>>>(
                bufs[slot].d_value, bufs[slot].d_eroded, width, height, kernelSize);
            dilateKernel<<<grid, block, morphShared, streams[slot]>>>(
                bufs[slot].d_eroded, bufs[slot].d_opened, width, height, kernelSize);

            // closing = erode(dilate(value))
            dilateKernel<<<grid, block, morphShared, streams[slot]>>>(
                bufs[slot].d_value, bufs[slot].d_dilated, width, height, kernelSize);
            erodeKernel<<<grid, block, morphShared, streams[slot]>>>(
                bufs[slot].d_dilated, bufs[slot].d_closed, width, height, kernelSize);

            // top hat + black hat
            maximizeContrastKernel<<<grid, block, 0, streams[slot]>>>(
                bufs[slot].d_value, bufs[slot].d_opened, bufs[slot].d_closed,
                bufs[slot].d_contrasted, width, height);

            // Gaussian blur 5x5
            gaussianBlurKernel<<<grid, block, blurShared, streams[slot]>>>(
                bufs[slot].d_contrasted, bufs[slot].d_blurred, width, height);

            // Adaptive threshold -> final binary image
            adaptiveThresholdKernel<<<grid, block, adaptShared, streams[slot]>>>(
                bufs[slot].d_blurred, d_output[slot], width, height, ADAPTIVE_C);

            cudaMemcpyAsync(imgGrayscaleBatch[i].data, bufs[slot].d_value, graySize, cudaMemcpyDeviceToHost, streams[slot]);
            cudaMemcpyAsync(imgThreshBatch[i].data, d_output[slot], graySize, cudaMemcpyDeviceToHost, streams[slot]);
        }

        for (int slot = 0; slot < waveCount; ++slot) {
            cudaStreamSynchronize(streams[slot]);
            cudaStreamDestroy(streams[slot]);
            cudaFree(d_input[slot]);
            cudaFree(d_output[slot]);
            freeGPUBuffers(bufs[slot]);
        }
    }
}