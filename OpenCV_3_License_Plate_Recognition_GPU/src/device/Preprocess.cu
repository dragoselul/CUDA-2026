// Preprocess.cu
// 6-kernel pipeline: BGR → V → morph (open/close) → contrast → blur → adaptive threshold.
// All kernels run asynchronously on the caller-supplied stream.

#include "Preprocess.cuh"
#include <cmath>

#define BLOCK_W 32
#define BLOCK_H 8

__constant__ float c_gaussKernel[25];    // 5×5 Gaussian weights
__constant__ float c_adaptiveKernel[225];// 15×15 adaptive threshold weights

void initKernelWeights()
{
    static bool done = false;
    if (done) return;
    done = true;

    // 5×5 Gaussian, σ=1
    {
        const int radius = 2;
        float host[25]; float sum = 0.f;
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) {
                float dy = ky - radius, dx = kx - radius;
                float w = expf(-(dx*dx + dy*dy) / 2.f);
                host[ky*5+kx] = w; sum += w;
            }
        for (int i = 0; i < 25; i++) host[i] /= sum;
        cudaMemcpyToSymbol(c_gaussKernel, host, sizeof(host));
    }

    // 15×15 adaptive threshold kernel
    {
        const int   radius = 7;
        const float sigma  = (15.f / 2.f - 1.f) * 0.3f + 0.8f;
        float host[225]; float sum = 0.f;
        for (int ky = 0; ky < 15; ky++)
            for (int kx = 0; kx < 15; kx++) {
                float dy = ky - radius, dx = kx - radius;
                float w = expf(-(dx*dx + dy*dy) / (2.f * sigma * sigma));
                host[ky*15+kx] = w; sum += w;
            }
        for (int i = 0; i < 225; i++) host[i] /= sum;
        cudaMemcpyToSymbol(c_adaptiveKernel, host, sizeof(host));
    }
}

__device__ static int clampDev(int v, int lo, int hi) { return max(lo, min(v, hi)); }

// ─── kernel 1: BGR → V (max channel) ─────────────────────────────────────────
__global__ void extractValueKernel(const unsigned char* in, unsigned char* out, int W, int H)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = (y*W+x)*3;
    out[y*W+x] = max(in[i], max(in[i+1], in[i+2]));
}

// ─── kernel 2: Gaussian blur 5×5 ─────────────────────────────────────────────
__global__ void gaussianBlurKernel(const unsigned char* in, unsigned char* out, int W, int H)
{
    const int radius = 2, tileW = BLOCK_W+4, tileH = BLOCK_H+4;
    extern __shared__ unsigned char s[];
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    for (int ty = threadIdx.y; ty < tileH; ty += BLOCK_H)
        for (int tx = threadIdx.x; tx < tileW; tx += BLOCK_W)
            s[ty*tileW+tx] = in[clampDev(blockIdx.y*BLOCK_H+ty-radius,0,H-1)*W
                               + clampDev(blockIdx.x*BLOCK_W+tx-radius,0,W-1)];
    __syncthreads();
    if (x >= W || y >= H) return;
    float acc = 0.f;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            acc += c_gaussKernel[(dy+radius)*5+(dx+radius)]
                 * s[(threadIdx.y+radius+dy)*tileW+(threadIdx.x+radius+dx)];
    out[y*W+x] = (unsigned char)clampDev((int)acc, 0, 255);
}

// ─── kernel 3: erosion (min filter, 3×3) ─────────────────────────────────────
__global__ void erodeKernel(const unsigned char* in, unsigned char* out, int W, int H)
{
    const int radius = 1, tileW = BLOCK_W+2, tileH = BLOCK_H+2;
    extern __shared__ unsigned char s[];
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    for (int ty = threadIdx.y; ty < tileH; ty += BLOCK_H)
        for (int tx = threadIdx.x; tx < tileW; tx += BLOCK_W)
            s[ty*tileW+tx] = in[clampDev(blockIdx.y*BLOCK_H+ty-radius,0,H-1)*W
                               + clampDev(blockIdx.x*BLOCK_W+tx-radius,0,W-1)];
    __syncthreads();
    if (x >= W || y >= H) return;
    unsigned char v = 255;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            v = min(v, s[(threadIdx.y+radius+dy)*tileW+(threadIdx.x+radius+dx)]);
    out[y*W+x] = v;
}

// ─── kernel 4: dilation (max filter, 3×3) ────────────────────────────────────
__global__ void dilateKernel(const unsigned char* in, unsigned char* out, int W, int H)
{
    const int radius = 1, tileW = BLOCK_W+2, tileH = BLOCK_H+2;
    extern __shared__ unsigned char s[];
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    for (int ty = threadIdx.y; ty < tileH; ty += BLOCK_H)
        for (int tx = threadIdx.x; tx < tileW; tx += BLOCK_W)
            s[ty*tileW+tx] = in[clampDev(blockIdx.y*BLOCK_H+ty-radius,0,H-1)*W
                               + clampDev(blockIdx.x*BLOCK_W+tx-radius,0,W-1)];
    __syncthreads();
    if (x >= W || y >= H) return;
    unsigned char v = 0;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            v = max(v, s[(threadIdx.y+radius+dy)*tileW+(threadIdx.x+radius+dx)]);
    out[y*W+x] = v;
}

// ─── kernel 5: top-hat + black-hat contrast ───────────────────────────────────
__global__ void maximizeContrastKernel(const unsigned char* d_value,
                                        const unsigned char* d_opened,
                                        const unsigned char* d_closed,
                                              unsigned char* d_out,
                                        int W, int H)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y*W+x;
    int16_t v  = d_value[i];
    int16_t r  = v + (v - (int16_t)d_opened[i]) - ((int16_t)d_closed[i] - v);
    d_out[i]   = (unsigned char)clampDev((int)r, 0, 255);
}

// ─── kernel 6: adaptive Gaussian threshold 15×15 ─────────────────────────────
__global__ void adaptiveThresholdKernel(const unsigned char* in, unsigned char* out,
                                         int W, int H, float C)
{
    const int radius = 7, tileW = BLOCK_W+14, tileH = BLOCK_H+14;
    extern __shared__ unsigned char s[];
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    for (int ty = threadIdx.y; ty < tileH; ty += BLOCK_H)
        for (int tx = threadIdx.x; tx < tileW; tx += BLOCK_W)
            s[ty*tileW+tx] = in[clampDev(blockIdx.y*BLOCK_H+ty-radius,0,H-1)*W
                               + clampDev(blockIdx.x*BLOCK_W+tx-radius,0,W-1)];
    __syncthreads();
    if (x >= W || y >= H) return;
    float mean = 0.f;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            mean += c_adaptiveKernel[(dy+radius)*15+(dx+radius)]
                  * s[(threadIdx.y+radius+dy)*tileW+(threadIdx.x+radius+dx)];
    float pixel = s[(threadIdx.y+radius)*tileW+(threadIdx.x+radius)];
    out[y*W+x] = (pixel < mean - C) ? 255 : 0;
}

// =============================================================================
// Buffer management
// =============================================================================
PreprocessBuffers allocPreprocessBuffers(int W, int H)
{
    initKernelWeights();
    PreprocessBuffers b;
    size_t sz = (size_t)W * H;
    cudaMalloc(&b.d_value,      sz);
    cudaMalloc(&b.d_eroded,     sz);
    cudaMalloc(&b.d_opened,     sz);
    cudaMalloc(&b.d_dilated,    sz);
    cudaMalloc(&b.d_closed,     sz);
    cudaMalloc(&b.d_contrasted, sz);
    cudaMalloc(&b.d_blurred,    sz);
    return b;
}

void freePreprocessBuffers(PreprocessBuffers& b)
{
    cudaFree(b.d_value);
    cudaFree(b.d_eroded);
    cudaFree(b.d_opened);
    cudaFree(b.d_dilated);
    cudaFree(b.d_closed);
    cudaFree(b.d_contrasted);
    cudaFree(b.d_blurred);
    b = {};
}

// =============================================================================
// preprocessDevice — async, no internal sync
// =============================================================================
void preprocessDevice(const unsigned char* d_bgr,
                      unsigned char*       d_thresh,
                      int                  W,
                      int                  H,
                      PreprocessBuffers&   bufs,
                      cudaStream_t         stream)
{
    initKernelWeights();

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W+BLOCK_W-1)/BLOCK_W, (H+BLOCK_H-1)/BLOCK_H);

    const size_t morphSh = (size_t)(BLOCK_W+2)*(BLOCK_H+2);
    const size_t blurSh  = (size_t)(BLOCK_W+4)*(BLOCK_H+4);
    const size_t adaptSh = (size_t)(BLOCK_W+14)*(BLOCK_H+14);

    extractValueKernel      <<<grid, block,       0, stream>>>(d_bgr, bufs.d_value, W, H);
    erodeKernel             <<<grid, block, morphSh, stream>>>(bufs.d_value,   bufs.d_eroded,  W, H);
    dilateKernel            <<<grid, block, morphSh, stream>>>(bufs.d_eroded,  bufs.d_opened,  W, H);
    dilateKernel            <<<grid, block, morphSh, stream>>>(bufs.d_value,   bufs.d_dilated, W, H);
    erodeKernel             <<<grid, block, morphSh, stream>>>(bufs.d_dilated, bufs.d_closed,  W, H);
    maximizeContrastKernel  <<<grid, block,       0, stream>>>(bufs.d_value, bufs.d_opened,
                                                               bufs.d_closed, bufs.d_contrasted, W, H);
    gaussianBlurKernel      <<<grid, block,  blurSh, stream>>>(bufs.d_contrasted, bufs.d_blurred, W, H);
    adaptiveThresholdKernel <<<grid, block, adaptSh, stream>>>(bufs.d_blurred, d_thresh, W, H, 2.0f);
    // No cudaStreamSynchronize — caller controls when to sync.
}
