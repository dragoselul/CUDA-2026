// CCL.cu
// GPU Connected Components Labeling using iterative parallel union-find.
//
// Kernels (executed in order):
//   1. cclInitKernel      — label[i] = i (foreground) or -1 (background)
//   2. cclPropagateKernel — each pixel atomicMin-merges roots with 4-neighbours
//                           repeat until convergence or maxIter
//   3. cclFlattenKernel   — path-compress every label to its root
//   4. cclCompactKernel   — assign dense IDs 0..N-1 to root pixels via atomicAdd
//   5. cclStatsKernel     — compute bbox + centroid per component
//   6. filterBlobsKernel  — keep only char-sized blobs; writes compact FilteredBlob[]
//                           via atomicAdd so only survivors cross PCIe

#include "CCL.cuh"
#include <cstdio>
#include <climits>

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

// ─── kernel 2: propagate ─────────────────────────────────────────────────────
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
// One thread per component. Applies char-size predicates; survivors are written
// to d_out via atomicAdd so only they cross PCIe (not all 4096 ComponentStats).
__global__ void filterBlobsKernel(const ComponentStats* d_stats, int numStats,
                                   FilteredBlob* d_out, int* d_count, int maxOut)
{
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

// ─── shared CCL setup / teardown ─────────────────────────────────────────────
struct CCLWorkspace {
    int32_t*        d_label;
    int32_t*        d_compactMap;
    int*            d_changed;
    int*            d_numComp;
    ComponentStats* d_stats;
};

static CCLWorkspace allocWorkspace(int N)
{
    CCLWorkspace ws;
    cudaMalloc(&ws.d_label,      N * sizeof(int32_t));
    cudaMalloc(&ws.d_compactMap, N * sizeof(int32_t));
    cudaMalloc(&ws.d_changed,        sizeof(int));
    cudaMalloc(&ws.d_numComp,        sizeof(int));
    cudaMalloc(&ws.d_stats,      CCL_MAX_COMPONENTS * sizeof(ComponentStats));
    return ws;
}

static void freeWorkspace(CCLWorkspace& ws)
{
    cudaFree(ws.d_label);
    cudaFree(ws.d_compactMap);
    cudaFree(ws.d_changed);
    cudaFree(ws.d_numComp);
    cudaFree(ws.d_stats);
}

static int runCCLKernels(const unsigned char* d_thresh, int W, int H,
                          CCLWorkspace& ws, cudaStream_t stream, int maxIter)
{
    const int N = W * H;
    dim3 block(CCL_BLOCK_W, CCL_BLOCK_H);
    dim3 grid((W + CCL_BLOCK_W - 1) / CCL_BLOCK_W,
              (H + CCL_BLOCK_H - 1) / CCL_BLOCK_H);

    cclInitStatsKernel<<<(CCL_MAX_COMPONENTS + 255) / 256, 256, 0, stream>>>(
        ws.d_stats, CCL_MAX_COMPONENTS);
    cudaMemsetAsync(ws.d_compactMap, 0xFF, N * sizeof(int32_t), stream);

    int zero = 0;
    cudaMemcpyAsync(ws.d_numComp, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);

    cclInitKernel<<<grid, block, 0, stream>>>(d_thresh, ws.d_label, W, H);

    // Convergence loop — must sync to read d_changed each iteration.
    // This is unavoidable: the loop cannot be pipelined across streams.
    cudaStreamSynchronize(stream);
    int h_changed = 1;
    for (int iter = 0; iter < maxIter && h_changed; iter++) {
        h_changed = 0;
        cudaMemcpy(ws.d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        cclPropagateKernel<<<grid, block, 0, stream>>>(ws.d_label, ws.d_changed, W, H);
        cudaMemcpy(&h_changed, ws.d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cclFlattenKernel <<<grid, block, 0, stream>>>(ws.d_label, W, H);
    cclCompactKernel <<<grid, block, 0, stream>>>(ws.d_label, ws.d_compactMap,
                                                   ws.d_numComp, W, H);
    int h_numComp = 0;
    cudaMemcpy(&h_numComp, ws.d_numComp, sizeof(int), cudaMemcpyDeviceToHost);
    cclStatsKernel<<<grid, block, 0, stream>>>(ws.d_label, ws.d_compactMap,
                                               ws.d_stats, W, H);
    return min(h_numComp, CCL_MAX_COMPONENTS);
}

// =============================================================================
// runCCLWithFilter — preferred public API
// =============================================================================
void runCCLWithFilter(
    const unsigned char* d_thresh,
    int                  width,
    int                  height,
    FilteredBlob*        d_filtered,
    int*                 d_num_filtered,
    FilteredBlob*        h_filtered,
    int*                 h_num_filtered,
    cudaStream_t         stream,
    int                  maxIter)
{
    CCLWorkspace ws = allocWorkspace(width * height);

    // Zero the filter counter on device before launching the filter kernel.
    cudaMemsetAsync(d_num_filtered, 0, sizeof(int), stream);

    int numComp = runCCLKernels(d_thresh, width, height, ws, stream, maxIter);

    // Filter on GPU — only char-sized blobs survive.
    const int filterBlocks = (numComp + 255) / 256;
    if (filterBlocks > 0)
        filterBlobsKernel<<<filterBlocks, 256, 0, stream>>>(
            ws.d_stats, numComp, d_filtered, d_num_filtered, CCL_MAX_FILTERED);

    // Async D2H — caller syncs when needed.
    cudaMemcpyAsync(h_num_filtered, d_num_filtered, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_filtered, d_filtered,
                    CCL_MAX_FILTERED * sizeof(FilteredBlob),
                    cudaMemcpyDeviceToHost, stream);

    freeWorkspace(ws);
}

// =============================================================================
// runCCL — legacy API, always synchronises
// =============================================================================
int runCCL(const unsigned char* d_thresh, int width, int height,
           ComponentStats* h_statsOut, int maxIter)
{
    CCLWorkspace ws = allocWorkspace(width * height);
    int numComp = runCCLKernels(d_thresh, width, height, ws, 0, maxIter);
    cudaMemcpy(h_statsOut, ws.d_stats, numComp * sizeof(ComponentStats),
               cudaMemcpyDeviceToHost);
    freeWorkspace(ws);
    return numComp;
}
