// KNN.cu
// GPU brute-force k=1 nearest neighbor.
// One CUDA block per query char; threads compute L2 to training samples in
// parallel; shared-memory tree reduction finds the nearest neighbour.

#include "KNN.cuh"
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#define BLOCK_KNN 192   // must be >= numSamples; power-of-2 × warp size

__global__ void knnKernel(
    const float*   d_queries,
    int            numQueries,
    const float*   d_training,
    const int32_t* d_trainLabels,
    int            numSamples,
    int32_t*       d_results)
{
    const int q   = blockIdx.x;
    const int tid = threadIdx.x;
    if (q >= numQueries) return;

    extern __shared__ char s_raw[];
    float* s_query = reinterpret_cast<float*>(s_raw);
    float* s_dist  = reinterpret_cast<float*>(s_raw + KNN_FEATURES * sizeof(float));
    int*   s_idx   = reinterpret_cast<int*>  (s_dist + BLOCK_KNN);

    for (int i = tid; i < KNN_FEATURES; i += blockDim.x)
        s_query[i] = d_queries[q * KNN_FEATURES + i];
    __syncthreads();

    float dist = FLT_MAX;
    if (tid < numSamples) {
        const float* train = d_training + tid * KNN_FEATURES;
        float acc = 0.f;
        for (int i = 0; i < KNN_FEATURES; i++) { float d = s_query[i]-train[i]; acc += d*d; }
        dist = acc;
    }
    s_dist[tid] = dist;
    s_idx[tid]  = tid;
    __syncthreads();

    for (int stride = BLOCK_KNN / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_dist[tid+stride] < s_dist[tid]) {
            s_dist[tid] = s_dist[tid+stride];
            s_idx[tid]  = s_idx[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int best = s_idx[0];
        d_results[q] = (best >= 0 && best < numSamples) ? d_trainLabels[best] : '?';
    }
}

// =============================================================================
bool loadKNNModel(const char* binPath, KNNModel& model)
{
    FILE* f = fopen(binPath, "rb");
    if (!f) { fprintf(stderr, "loadKNNModel: cannot open %s\n", binPath); return false; }

    uint32_t header[3];
    if (fread(header, sizeof(uint32_t), 3, f) != 3) { fclose(f); return false; }
    const int N = (int)header[0], F = (int)header[1];
    if (F != KNN_FEATURES) {
        fprintf(stderr, "loadKNNModel: expected %d features, got %d\n", KNN_FEATURES, F);
        fclose(f); return false;
    }

    auto* h_labels   = new int32_t[N];
    auto* h_training = new float[N * F];
    bool ok = fread(h_labels,   sizeof(int32_t), N,     f) == (size_t)N
           && fread(h_training, sizeof(float),   N * F, f) == (size_t)(N * F);
    fclose(f);
    if (!ok) { delete[] h_labels; delete[] h_training; return false; }

    cudaMalloc(&model.d_trainLabels, N * sizeof(int32_t));
    cudaMalloc(&model.d_training,    N * F * sizeof(float));
    cudaMemcpy(model.d_trainLabels, h_labels,   N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(model.d_training,    h_training, N * F * sizeof(float), cudaMemcpyHostToDevice);
    model.numSamples = N;
    delete[] h_labels; delete[] h_training;
    printf("KNN model loaded: %d samples × %d features\n", N, F);
    return true;
}

void freeKNNModel(KNNModel& model)
{
    cudaFree(model.d_training);    model.d_training    = nullptr;
    cudaFree(model.d_trainLabels); model.d_trainLabels = nullptr;
    model.numSamples = 0;
}

// =============================================================================
// runKNNDevice — fully async, no internal sync or allocation
// =============================================================================
void runKNNDevice(const KNNModel& model,
                  const float*    d_queries,
                  int             numChars,
                  int32_t*        d_results,
                  int32_t*        h_labels,
                  cudaStream_t    stream)
{
    if (numChars <= 0) return;

    const size_t sharedBytes =
        KNN_FEATURES * sizeof(float) +
        BLOCK_KNN    * sizeof(float) +
        BLOCK_KNN    * sizeof(int);

    knnKernel<<<numChars, BLOCK_KNN, sharedBytes, stream>>>(
        d_queries, numChars,
        model.d_training, model.d_trainLabels, model.numSamples,
        d_results);

    cudaMemcpyAsync(h_labels, d_results, numChars * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
}
