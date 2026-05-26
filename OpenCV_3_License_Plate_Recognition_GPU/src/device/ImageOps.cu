// ImageOps.cu
// GPU image geometry kernels: bilinear resize, char-ROI resize, affine warp+crop.

#include "ImageOps.cuh"
#include <cmath>

#define RS_BLOCK_W 32
#define RS_BLOCK_H 8
#define WC_BLOCK_W 32
#define WC_BLOCK_H 8

// =============================================================================
// Bilinear resize — single channel
// =============================================================================
__global__ static void resizeBilinearKernel(
    const unsigned char* d_src, int srcW, int srcH,
    unsigned char*       d_dst, int dstW, int dstH)
{
    int ox = blockIdx.x * RS_BLOCK_W + threadIdx.x;
    int oy = blockIdx.y * RS_BLOCK_H + threadIdx.y;
    if (ox >= dstW || oy >= dstH) return;

    float sx = (ox + 0.5f) * ((float)srcW / dstW) - 0.5f;
    float sy = (oy + 0.5f) * ((float)srcH / dstH) - 0.5f;
    int x0 = max(0, min((int)floorf(sx), srcW-1)), x1 = min(x0+1, srcW-1);
    int y0 = max(0, min((int)floorf(sy), srcH-1)), y1 = min(y0+1, srcH-1);
    float fx = sx - floorf(sx), fy = sy - floorf(sy);

    float top    = d_src[y0*srcW+x0] + fx*(d_src[y0*srcW+x1]-d_src[y0*srcW+x0]);
    float bottom = d_src[y1*srcW+x0] + fx*(d_src[y1*srcW+x1]-d_src[y1*srcW+x0]);
    d_dst[oy*dstW+ox] = (unsigned char)max(0.f, min(255.f, top + fy*(bottom-top)));
}

void runResizeInto(const unsigned char* d_src, int srcW, int srcH,
                   unsigned char* d_dst, int dstW, int dstH, cudaStream_t stream)
{
    dim3 block(RS_BLOCK_W, RS_BLOCK_H);
    dim3 grid((dstW+RS_BLOCK_W-1)/RS_BLOCK_W, (dstH+RS_BLOCK_H-1)/RS_BLOCK_H);
    resizeBilinearKernel<<<grid, block, 0, stream>>>(d_src, srcW, srcH, d_dst, dstW, dstH);
}

// =============================================================================
// Char ROI resize — one block per char, outputs float32 for KNN
// =============================================================================
__global__ static void charROIResizeKernel(
    const unsigned char* d_thresh, int W, int H,
    const Rect2i* d_rects, int numRects,
    float* d_queries)
{
    const int ci = blockIdx.x, ox = threadIdx.x, oy = threadIdx.y;
    if (ci >= numRects) return;
    const Rect2i r = d_rects[ci];

    float sx = (ox + 0.5f) * ((float)r.width  / KNN_CHAR_W) - 0.5f;
    float sy = (oy + 0.5f) * ((float)r.height / KNN_CHAR_H) - 0.5f;

    int x0 = max(0, min((int)floorf(sx), r.width -1)), x1 = min(x0+1, r.width -1);
    int y0 = max(0, min((int)floorf(sy), r.height-1)), y1 = min(y0+1, r.height-1);
    float fx = sx - floorf(sx), fy = sy - floorf(sy);

    int gx0 = min(max(r.x+x0,0),W-1), gx1 = min(max(r.x+x1,0),W-1);
    int gy0 = min(max(r.y+y0,0),H-1), gy1 = min(max(r.y+y1,0),H-1);

    float top    = d_thresh[gy0*W+gx0] + fx*(d_thresh[gy0*W+gx1]-d_thresh[gy0*W+gx0]);
    float bottom = d_thresh[gy1*W+gx0] + fx*(d_thresh[gy1*W+gx1]-d_thresh[gy1*W+gx0]);
    d_queries[ci*KNN_FEATURES + oy*KNN_CHAR_W + ox] = top + fy*(bottom-top);
}

void runCharROIResize(const unsigned char* d_thresh, int W, int H,
                      const Rect2i* h_rects, int numRects,
                      Rect2i* d_rects_buf, float* d_queries,
                      cudaStream_t stream)
{
    if (numRects <= 0) return;
    cudaMemcpyAsync(d_rects_buf, h_rects, numRects * sizeof(Rect2i),
                    cudaMemcpyHostToDevice, stream);
    dim3 block(KNN_CHAR_W, KNN_CHAR_H);
    charROIResizeKernel<<<numRects, block, 0, stream>>>(
        d_thresh, W, H, d_rects_buf, numRects, d_queries);
}

// =============================================================================
// Affine warp + crop — BGR, bilinear
// =============================================================================
__global__ static void warpCropKernel(
    const unsigned char* d_src, int W_src, int H_src,
    unsigned char*       d_dst, int outW,  int outH,
    float cx, float cy, float angleDeg)
{
    int ox = blockIdx.x * WC_BLOCK_W + threadIdx.x;
    int oy = blockIdx.y * WC_BLOCK_H + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const float a  = angleDeg * (3.14159265358979f / 180.f);
    const float ca = cosf(a), sa = sinf(a);
    float dx = ox - outW * 0.5f, dy = oy - outH * 0.5f;
    float sx = cx + dx*ca + dy*sa;
    float sy = cy - dx*sa + dy*ca;

    int x0 = max(0, min((int)floorf(sx), W_src-1)), x1 = min(x0+1, W_src-1);
    int y0 = max(0, min((int)floorf(sy), H_src-1)), y1 = min(y0+1, H_src-1);
    float fx = sx - floorf(sx), fy = sy - floorf(sy);

    int out_idx = (oy * outW + ox) * 3;
    for (int c = 0; c < 3; c++) {
        float v00 = d_src[(y0*W_src+x0)*3+c], v10 = d_src[(y0*W_src+x1)*3+c];
        float v01 = d_src[(y1*W_src+x0)*3+c], v11 = d_src[(y1*W_src+x1)*3+c];
        float top = v00 + fx*(v10-v00), bot = v01 + fx*(v11-v01);
        d_dst[out_idx+c] = (unsigned char)max(0.f, min(255.f, top + fy*(bot-top)));
    }
}

void runBatchWarpCrop(const unsigned char* d_src, int W_src, int H_src,
                      const WarpParams* params, unsigned char* const* d_dsts,
                      int count, cudaStream_t stream)
{
    dim3 block(WC_BLOCK_W, WC_BLOCK_H);
    for (int i = 0; i < count; i++) {
        const WarpParams& p = params[i];
        dim3 grid((p.outW+WC_BLOCK_W-1)/WC_BLOCK_W, (p.outH+WC_BLOCK_H-1)/WC_BLOCK_H);
        warpCropKernel<<<grid, block, 0, stream>>>(
            d_src, W_src, H_src, d_dsts[i],
            p.outW, p.outH, p.cx, p.cy, p.angleDeg);
    }
}
