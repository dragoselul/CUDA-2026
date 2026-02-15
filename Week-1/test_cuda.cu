#include <stdio.h>

// Kernel definition [9]
__global__ void vecadd(int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = i; // Simple dummy operation
    }
}

int main() {
    int n = 256;
    int *d_c;
    int size = n * sizeof(int);

    // 1. Allocate Device Memory 
    cudaMalloc((void**)&d_c, size);

    // 2. Launch Kernel [6]
    // Launching 1 block with 256 threads
    vecadd<<<1, 256>>>(d_c, n);

    // 3. Synchronize [6]
    cudaDeviceSynchronize();

    // 4. Cleanup 
    cudaFree(d_c);

    printf("Success! CUDA is set up.\n");
    return 0;
}