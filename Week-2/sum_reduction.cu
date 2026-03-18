#include "sum_reduction.cuh"

__global__ void reduce0_kernel(const int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < (unsigned int)n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if ((tid % (2 * s) == 0) && (tid + s < blockDim.x)) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

extern "C" void launch_reduce0(int blocks, int threads, const int *d_in, int *d_out, int n) {
    size_t sharedMemSize = threads * sizeof(int);
    reduce0_kernel<<<blocks, threads, sharedMemSize>>>(d_in, d_out, n);
}