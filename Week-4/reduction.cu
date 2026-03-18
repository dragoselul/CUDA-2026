#include "reduction.cuh"

// VERSION 0: Bare Minimum
__global__ void g_reduce0(int n, const int* d_in, int* d_out) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Use a local copy to avoid modifying d_in directly if it's const
    // (Note: Real reductions usually load into shared mem or registers immediately)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && i + s < n) {
            ((int*)d_in)[i] += d_in[i + s]; 
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = d_in[i];
}

// VERSION 1: Shared Memory
__global__ void g_reduce1(int n, const int* d_in, int* d_out) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_in[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
    // We 'flatten' the tray access so the compiler doesn't have to calculate 's'
    volatile int* smem = sdata; 

    // We merge the 'top half' of the warp into the 'bottom half'
    smem[tid] += smem[tid + 32]; // Merge 64 into 32 
    smem[tid] += smem[tid + 16]; // Merge 32 into 16
    smem[tid] += smem[tid + 8];  // Merge 16 into 8
    smem[tid] += smem[tid + 4];  // Merge 8 into 4
    smem[tid] += smem[tid + 2];  // Merge 4 into 2
    smem[tid] += smem[tid + 1];  // Merge 2 into 1 (The Winner!)
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


// Warp shuffle
__global__ void g_reduce2(int n, const int* d_in, int* d_out) {
    // We still need a tiny bit of shared memory for the Warp Leaders to talk
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Initial Load
    int val = (i < n) ? d_in[i] : 0;

    // 2. Warp-Level Reduction
    // Every warp (32 threads) reduces its own values internally
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 3. Logistics: Moving Warp totals to Shared Memory
    // Only the 'Leader' of each warp (tid 0, 32, 64...) writes to the tray
    int warpId = tid / 32;
    int laneId = tid % 32; // Which robot am I in my warp?

    if (laneId == 0) sdata[warpId] = val;

    __syncthreads(); // Wait for all warp leaders to report in

    // 4. Final Final: The last Warp finishes the block
    // If we had 256 threads, we now have 8 warp totals in sdata.
    if (warpId == 0) {
        // Read the warp totals (if they exist, else 0)
        val = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0;

        // One last shuffle tournament for the warp leaders
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        
        // Final winner writes to global output
        if (tid == 0) d_out[blockIdx.x] = val;
    }
}

// Grid stride + warp shuffle
__global__ void g_reduce3(int n, const int* d_in, int* d_out) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    int val = 0;

    // --- GRID STRIDE LOOP ---
    // Instead of one robot per item, each robot walks down the belt
    // and sums items until the end. This is MUCH faster for 1GB of data.
    while (i < n) {
        val += d_in[i];
        i += gridSize;
    }

    // --- WARP SHUFFLE REDUCTION ---
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    int warpId = tid / 32;
    int laneId = tid % 32;
    if (laneId == 0) sdata[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) d_out[blockIdx.x] = val;
    }
}