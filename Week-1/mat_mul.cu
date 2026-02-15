#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024       // 1024x1024 matrix
#define BLOCK_SIZE 16 // 16x16 threads per block

// 1. CPU OpenMP Kernel
void matrixMulCPU(float *A, float *B, float *C) {
    // This tells the CPU to divide the nested loops across all available CPU cores
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. GPU CUDA Kernel
__global__ void matrixMulGPU(float *A, float *B, float *C) {
    // Each thread calculates exactly one element in the C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);
    
    // Allocate Host (CPU) Memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    // Initialize matrices with dummy values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("Executing on CPU (OpenMP)...\n");
    double start_cpu = omp_get_wtime();
    matrixMulCPU(h_A, h_B, h_C_cpu);
    double end_cpu = omp_get_wtime();
    printf("CPU Time: %f seconds\n\n", end_cpu - start_cpu);

    printf("Executing on GPU (CUDA)...\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Setup 2D Grid and 2D Blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warmup launch (to avoid timing the GPU wake-up overhead)
    matrixMulGPU<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Time the CUDA execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGPU<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    printf("GPU Time: %f seconds\n", milliseconds / 1000.0);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);

    return 0;
}