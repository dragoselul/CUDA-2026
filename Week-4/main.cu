#include <iostream>
#include <cuda_runtime.h>
#include "reduction.cuh"

#define SPLITS 8

__global__ void g_warmup(int *d_in)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_in[0] = 0;
    }
}

int main()
{
    // --- STEP 1: Define the Problem ---
    // 1 << 28 is ~268 million elements (1GB of ints)
    int n = 1 << 28;
    int threadsPerBlock = 256;

    // Calculate how many elements each stream "belt" handles
    int chunkSize = n / SPLITS;
    // Instead of blocksPerChunk = (chunkSize + threads - 1) / threads;
    // Use a fixed "Sweet Spot" for the Hopper GPU (e.g., 1024 or 2048)
    int blocksPerChunk = 160;
    ;
    int totalBlocks = blocksPerChunk * SPLITS;

    size_t sizeIn = n * sizeof(int);
    size_t sizeOut = totalBlocks * sizeof(int);

    // --- STEP 2: Pinned Host Memory (The High-Speed Loading Dock) ---
    // We use cudaHostAlloc so the DMA engine can access this without CPU help.
    int *h_in, *h_out;
    cudaHostAlloc(&h_in, sizeIn, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, sizeOut, cudaHostAllocDefault);

    // Fill with 1s
    for (int i = 0; i < n; i++)
        h_in[i] = 1;

    // --- STEP 3: Device Memory (The Garage Floor) ---
    int *d_in, *d_out;
    cudaMalloc(&d_in, sizeIn);
    cudaMalloc(&d_out, sizeOut);

    // --- STEP 4: Create Logistics Channels (Streams) ---
    cudaStream_t streams[SPLITS];
    for (int s = 0; s < SPLITS; ++s)
    {
        cudaStreamCreate(&streams[s]);
    }

    // Warm up the GPU

    g_warmup<<<1, 1>>>(d_in);
    cudaDeviceSynchronize(); // Force the driver to finish all "hidden" setup

    // --- STEP 5: The Overlap Loop (Pipelining) ---
    // Shared Memory size: 1 int per Warp in the block
    size_t smemSize = (threadsPerBlock / 32) * sizeof(int);

    for (int i = 0; i < SPLITS; i++)
    {
        int offsetIn = i * chunkSize;
        int offsetOut = i * blocksPerChunk;

        // 1. Send data for this chunk (Non-blocking)
        cudaMemcpyAsync(d_in + offsetIn, h_in + offsetIn,
                        chunkSize * sizeof(int),
                        cudaMemcpyHostToDevice, streams[i]);

        // 2. Launch Kernel for this chunk (Non-blocking)
        // Note: d_out + offsetOut ensures each chunk writes to a unique spot
        g_reduce3<<<blocksPerChunk, threadsPerBlock, smemSize, streams[i]>>>(chunkSize, d_in + offsetIn, d_out + offsetOut);

        // 3. Bring partial results back (Non-blocking)
        cudaMemcpyAsync(h_out + offsetOut, d_out + offsetOut,
                        blocksPerChunk * sizeof(int),
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // --- STEP 6: The "Checkered Flag" (Synchronization) ---
    // We wait here until all 8 streams have finished their pipeline
    cudaDeviceSynchronize();

    // --- STEP 7: Final Tally (Grace CPU finishes the job) ---
    long long totalSum = 0;
    for (int i = 0; i < totalBlocks; i++)
    {
        totalSum += h_out[i];
    }

    std::cout << "Final Sum: " << totalSum << " (Expected: " << n << ")" << std::endl;

    // --- STEP 8: Cleanup ---
    for (int s = 0; s < SPLITS; ++s)
        cudaStreamDestroy(streams[s]);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}