// OtsuThreshold.cu
// GPU histogram + CPU Otsu + GPU binary threshold.

#include "OtsuThreshold.cuh"
#include <cstring>

__global__ void histogramKernel(
    const unsigned char* d_input,
    int                  numPixels,
    int*                 d_hist)
{
    __shared__ int s_hist[256];
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; idx < numPixels; idx += stride) {
        atomicAdd(&s_hist[d_input[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&d_hist[threadIdx.x], s_hist[threadIdx.x]);
    }
}

__global__ void applyThresholdKernel(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  numPixels,
    unsigned char        thresh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numPixels; idx += stride) {
        d_output[idx] = (d_input[idx] >= thresh) ? 255 : 0;
    }
}

unsigned char computeOtsuThreshold(const int* h_hist, int numPixels) {
    if (numPixels <= 0) return 128;

    double totalMean = 0.0;
    for (int i = 0; i < 256; i++) totalMean += i * h_hist[i];
    totalMean /= numPixels;

    double w0 = 0.0, mean0 = 0.0;
    double bestVar = -1.0;
    unsigned char bestThresh = 0;

    for (int thresh = 0; thresh < 256; thresh++) {
        w0 += (double)h_hist[thresh] / numPixels;
        double w1 = 1.0 - w0;
        if (w0 < 1e-10 || w1 < 1e-10) continue;

        mean0 += (double)(thresh * h_hist[thresh]) / numPixels;
        double u0 = mean0 / w0;
        double u1 = (totalMean - mean0) / w1;

        double interClassVar = w0 * w1 * (u0 - u1) * (u0 - u1);
        if (interClassVar > bestVar) {
            bestVar    = interClassVar;
            bestThresh = (unsigned char)thresh;
        }
    }
    return bestThresh;
}

void runOtsuThreshold(
    const unsigned char* d_input,
    unsigned char*       d_output,
    int                  width,
    int                  height,
    cudaStream_t         stream)
{
    const int numPixels = width * height;

    int* d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemsetAsync(d_hist, 0, 256 * sizeof(int), stream);

    const int histBlocks = (numPixels + 255) / 256;
    histogramKernel<<<histBlocks, 256, 0, stream>>>(d_input, numPixels, d_hist);

    int h_hist[256];
    cudaMemcpyAsync(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned char thresh = computeOtsuThreshold(h_hist, numPixels);

    const int threshBlocks = (numPixels + 255) / 256;
    applyThresholdKernel<<<threshBlocks, 256, 0, stream>>>(d_input, d_output, numPixels, thresh);

    cudaFree(d_hist);
}
