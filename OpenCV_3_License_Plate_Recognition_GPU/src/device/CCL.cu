// CCL.cu
// GPU Connected Components Labeling using iterative parallel union-find.
//
// Kernels (executed in order):
//   1. cclInitKernel           — label[i] = i (foreground) or -1 (background)
//   2. cclPropagateCoopKernel  — cooperative grid-stride propagation; runs all
//                                iterations in one launch using grid.sync() to
//                                check convergence without round-tripping to host
//   3. cclFlattenKernel        — path-compress every label to its root
//   4. cclCompactKernel        — assign dense IDs 0..N-1 to root pixels
//   5. cclStatsKernel          — compute bbox + pixel count per component
//   6. filterBlobsKernel       — keep only char-sized blobs; reads d_numComp on
//                                device (same stream, no host sync needed), writes
//                                compact FilteredBlob[] to managed m_filtered

#include "CCL.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <climits>

namespace cg = cooperative_groups;

#define CCL_BLOCK_W 32
#define CCL_BLOCK_H 8

// ─── char filter thresholds (mirror DetectChars constants) ───────────────────
static constexpr int   FILT_MIN_AREA   = 80;
static constexpr int   FILT_MIN_WIDTH  = 2;
static constexpr int   FILT_MIN_HEIGHT = 8;
static constexpr float FILT_MIN_AR     = 0.25f;
static constexpr float FILT_MAX_AR     = 1.00f;

__device__ static int findRoot(const int32_t* label, int idx) {
    while (label[idx] != idx) idx = label[idx];
    return idx;
}

// ─── kernel 1: init ──────────────────────────────────────────────────────────
__global__ void cclInitKernel(const unsigned char* d_thresh, int32_t* d_label, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    d_label[i] = (d_thresh[i] > 0) ? i : -1;
}

// ─── kernel 2a: fallback propagate (non-cooperative, used when coop unavailable)
__global__ void cclPropagateKernel(int32_t* d_label, int* d_changed, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    if (d_label[i] < 0) return;

    int rp = findRoot(d_label, i);
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = { 0, 0,-1, 1};

    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        int j = ny * W + nx;
        if (d_label[j] < 0) continue;
        int rq = findRoot(d_label, j);
        if (rp != rq) {
            int lo = min(rp, rq), hi = max(rp, rq);
            atomicMin(&d_label[hi], lo);
            *d_changed = 1;
            rp = lo;
        }
    }
}

// ─── kernel 2b: cooperative grid-stride propagate ────────────────────────────
// One cooperative launch replaces the host-side convergence loop.
// grid.sync() provides the per-iteration barrier without round-tripping to host.
__global__ void cclPropagateCoopKernel(int32_t* d_label, int* d_changed,
                                        int N, int W, int H, int maxIter)
{
    cg::grid_group grid = cg::this_grid();
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    const int dx[4] = {-1, 1,  0, 0};
    const int dy[4] = { 0, 0, -1, 1};

    for (int iter = 0; iter < maxIter; iter++) {
        // Sync 1: reset flag — all threads see the write before propagating.
        if (tid == 0) *d_changed = 0;
        grid.sync();

        for (int i = tid; i < N; i += stride) {
            int x = i % W, y = i / W;
            if (d_label[i] < 0) continue;
            int rp = findRoot(d_label, i);
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int j = ny * W + nx;
                if (d_label[j] < 0) continue;
                int rq = findRoot(d_label, j);
                if (rp != rq) {
                    int lo = min(rp, rq), hi = max(rp, rq);
                    atomicMin(&d_label[hi], lo);
                    atomicOr(d_changed, 1);
                    rp = lo;
                }
            }
        }
        // Sync 2: all propagation writes to *d_changed are visible.
        grid.sync();

        // Cache into a register BEFORE sync 3: without a third sync, the
        // fastest block could race to the next iteration's reset (writing
        // *d_changed=0) before slower blocks have read it here, causing
        // divergent break decisions and a grid.sync() deadlock.
        bool converged = (*d_changed == 0);

        // Sync 3: all threads have read *d_changed into their local register.
        // No thread can proceed to the next iteration's reset until every
        // thread has passed this barrier — preventing the write-before-read race.
        grid.sync();

        if (converged) break;
    }
}

// ─── kernel 3: flatten ───────────────────────────────────────────────────────
__global__ void cclFlattenKernel(int32_t* d_label, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    if (d_label[i] >= 0) d_label[i] = findRoot(d_label, i);
}

// ─── kernel 4: compact ───────────────────────────────────────────────────────
__global__ void cclCompactKernel(const int32_t* d_label, int32_t* d_compactMap,
                                  int* d_numComp, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    if (d_label[i] != i) return;
    int cid = atomicAdd(d_numComp, 1);
    if (cid < CCL_MAX_COMPONENTS) d_compactMap[i] = cid;
}

// ─── kernel 5: stats ─────────────────────────────────────────────────────────
__global__ void cclStatsKernel(const int32_t* d_label, const int32_t* d_compactMap,
                                ComponentStats* d_stats, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i   = y * W + x;
    int lbl = d_label[i];
    if (lbl < 0) return;
    int cid = d_compactMap[lbl];
    if (cid < 0 || cid >= CCL_MAX_COMPONENTS) return;
    ComponentStats* s = d_stats + cid;
    atomicMin(&s->xMin, x); atomicMax(&s->xMax, x);
    atomicMin(&s->yMin, y); atomicMax(&s->yMax, y);
    atomicAdd(&s->pixelCount, 1);
    atomicAdd(&s->sumX, x);  atomicAdd(&s->sumY, y);
}

// ─── kernel: init stats array ────────────────────────────────────────────────
__global__ void cclInitStatsKernel(ComponentStats* d_stats, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_stats[i] = { INT_MAX, INT_MAX, 0, 0, 0, 0, 0 };
}

// ─── kernel 6: filter blobs → compact FilteredBlob[] ─────────────────────────
// Launched with a fixed CCL_MAX_COMPONENTS-wide grid; reads d_numComp from the
// device pointer (same stream as cclCompactKernel, so ordering is guaranteed).
// Writes directly to managed m_filtered so the CPU can read after stream sync
// with no explicit D2H copy.
__global__ void filterBlobsKernel(const ComponentStats* d_stats, const int* d_numComp,
                                   FilteredBlob* d_out, int* d_count, int maxOut)
{
    int numStats = min(*d_numComp, CCL_MAX_COMPONENTS);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numStats) return;

    const ComponentStats& s = d_stats[i];
    int w = s.xMax - s.xMin + 1;
    int h = s.yMax - s.yMin + 1;

    if (s.pixelCount  <= FILT_MIN_AREA)   return;
    if (w             <= FILT_MIN_WIDTH)   return;
    if (h             <  FILT_MIN_HEIGHT)  return;
    float ar = (float)w / (float)h;
    if (ar <= FILT_MIN_AR || ar >= FILT_MAX_AR) return;

    int slot = atomicAdd(d_count, 1);
    if (slot < maxOut)
        d_out[slot] = { s.xMin, s.yMin, w, h, i };
}

// =============================================================================
// CCLWorkspace alloc / free (exposed in CCL.cuh so PipelineContext pre-allocates)
// =============================================================================
CCLWorkspace allocWorkspace(int N)
{
    CCLWorkspace ws = {};
    cudaMalloc(&ws.d_label,      (size_t)N * sizeof(int32_t));
    cudaMalloc(&ws.d_compactMap, (size_t)N * sizeof(int32_t));
    cudaMalloc(&ws.d_changed,    sizeof(int));
    cudaMalloc(&ws.d_numComp,    sizeof(int));
    cudaMalloc(&ws.d_stats,      CCL_MAX_COMPONENTS * sizeof(ComponentStats));
    return ws;
}

void freeWorkspace(CCLWorkspace& ws)
{
    cudaFree(ws.d_label);
    cudaFree(ws.d_compactMap);
    cudaFree(ws.d_changed);
    cudaFree(ws.d_numComp);
    cudaFree(ws.d_stats);
    ws = {};
}

// =============================================================================
// runCCLKernels — internal; fully async on `stream` when coop launch succeeds
// =============================================================================
static void runCCLKernels(const unsigned char* d_thresh, int W, int H,
                           CCLWorkspace& ws, cudaStream_t stream, int maxIter,
                           bool allowCoop)
{
    const int N = W * H;
    dim3 block2d(CCL_BLOCK_W, CCL_BLOCK_H);
    dim3 grid2d((W + CCL_BLOCK_W - 1) / CCL_BLOCK_W,
                (H + CCL_BLOCK_H - 1) / CCL_BLOCK_H);

    cclInitStatsKernel<<<(CCL_MAX_COMPONENTS + 255) / 256, 256, 0, stream>>>(
        ws.d_stats, CCL_MAX_COMPONENTS);
    cudaMemsetAsync(ws.d_compactMap, 0xFF, (size_t)N * sizeof(int32_t), stream);
    cudaMemsetAsync(ws.d_numComp,    0,    sizeof(int), stream);

    cclInitKernel<<<grid2d, block2d, 0, stream>>>(d_thresh, ws.d_label, W, H);

    // ── Try cooperative propagate (eliminates host-side convergence loop) ──────
    // Only safe when this stream is the sole compute stream on the device;
    // cooperative launch fills all SMs and deadlocks if concurrent streams also
    // try to fill all SMs at the same time.
    int dev = 0;
    cudaGetDevice(&dev);
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

    bool usedCoop = false;
    if (allowCoop && supportsCoopLaunch) {
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM, cclPropagateCoopKernel, 256, 0);
        int coopBlocks = numSMs * maxBlocksPerSM;

        if (coopBlocks > 0) {
            void* args[] = {
                (void*)&ws.d_label,
                (void*)&ws.d_changed,
                (void*)&N,
                (void*)&W,
                (void*)&H,
                (void*)&maxIter
            };
            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)cclPropagateCoopKernel,
                dim3(coopBlocks), dim3(256),
                args, 0, stream);
            usedCoop = (err == cudaSuccess);
        }
    }

    if (!usedCoop) {
        // Fallback: original host-side sync loop.
        cudaStreamSynchronize(stream);
        int h_changed = 1;
        for (int iter = 0; iter < maxIter && h_changed; iter++) {
            h_changed = 0;
            cudaMemcpy(ws.d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
            cclPropagateKernel<<<grid2d, block2d, 0, stream>>>(ws.d_label, ws.d_changed, W, H);
            cudaMemcpy(&h_changed, ws.d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    cclFlattenKernel <<<grid2d, block2d, 0, stream>>>(ws.d_label, W, H);
    cclCompactKernel <<<grid2d, block2d, 0, stream>>>(ws.d_label, ws.d_compactMap,
                                                       ws.d_numComp, W, H);
    cclStatsKernel   <<<grid2d, block2d, 0, stream>>>(ws.d_label, ws.d_compactMap,
                                                       ws.d_stats, W, H);
    // filterBlobsKernel reads ws.d_numComp on the same stream — no host sync needed.
}

// =============================================================================
// runCCLWithFilter — preferred public API
// =============================================================================
void runCCLWithFilter(
    const unsigned char* d_thresh,
    int                  width,
    int                  height,
    CCLWorkspace&        ws,
    FilteredBlob*        d_filtered,
    int*                 d_num_filtered,
    FilteredBlob*        h_filtered,
    int*                 h_num_filtered,
    cudaStream_t         stream,
    int                  maxIter,
    bool                 allowCoop)
{
    cudaMemsetAsync(d_num_filtered, 0, sizeof(int), stream);

    runCCLKernels(d_thresh, width, height, ws, stream, maxIter, allowCoop);

    filterBlobsKernel<<<(CCL_MAX_COMPONENTS + 255) / 256, 256, 0, stream>>>(
        ws.d_stats, ws.d_numComp, d_filtered, d_num_filtered, CCL_MAX_FILTERED);

    // Async D2H into pinned host — fast DMA; caller syncs before reading.
    cudaMemcpyAsync(h_num_filtered, d_num_filtered, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_filtered, d_filtered,
                    CCL_MAX_FILTERED * sizeof(FilteredBlob),
                    cudaMemcpyDeviceToHost, stream);
}

// =============================================================================
// runCCL — legacy API, always synchronises
// =============================================================================
int runCCL(const unsigned char* d_thresh, int width, int height,
           ComponentStats* h_statsOut, int maxIter)
{
    CCLWorkspace ws = allocWorkspace(width * height);

    cudaMemsetAsync(ws.d_numComp, 0, sizeof(int), 0);
    runCCLKernels(d_thresh, width, height, ws, 0, maxIter, /*allowCoop=*/false);

    int h_numComp = 0;
    cudaMemcpy(&h_numComp, ws.d_numComp, sizeof(int), cudaMemcpyDeviceToHost);
    h_numComp = min(h_numComp, CCL_MAX_COMPONENTS);
    cudaMemcpy(h_statsOut, ws.d_stats, (size_t)h_numComp * sizeof(ComponentStats),
               cudaMemcpyDeviceToHost);

    freeWorkspace(ws);
    return h_numComp;
}
