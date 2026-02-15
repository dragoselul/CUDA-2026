#include <stdio.h>

// 1. The Kernel (Runs on the GPU)
// We pass in the input array, the output array, and the total size.
__global__ void multiplyByTwoStride(int *d_in, int *d_out, int num_elements) {
    // 1. Where do I start?
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. How big is my leap? (Total threads in the grid)
    int stride = blockDim.x * gridDim.x;

    // 3. The Loop
    // Thread 0 handles index 0, then leaps to 0 + stride, then 0 + 2*stride, etc.
    for (int i = index; i < num_elements; i += stride) {
        d_out[i] = d_in[i] * 2;
    }
}

int main() {
    int n = 100;
    int size = n * sizeof(int);

    // 2. Host Arrays (CPU RAM)
    int h_in[100];   // To hold our initial numbers (1 to 100)
    int h_out[100];  // To hold the results coming back from the GPU

    // Initialize the host input array with numbers 1 to 100
    for (int i = 0; i < n; i++) {
        h_in[i] = i + 1; 
    }

    // 3. Device Arrays (GPU VRAM)
    int *d_in;
    int *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // 4. Copy data from Host to Device
    // Send our 1-100 array over the PCIe bus to the GPU's memory
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 5. Launch the Kernel
    // We want 100 operations, so we launch 1 block containing 100 threads.
    multiplyByTwoStride<<<1, 20>>>(d_in, d_out, n);

    // 6. Copy data from Device back to Host
    // Bring the mathematically altered array back to the CPU so we can see it
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 7. Verify the results (Let's just print the first and last few to check)
    printf("First 3 elements:\n");
    for (int i = 0; i < 3; i++) {
        printf("Index %d: %d * 2 = %d\n", i, h_in[i], h_out[i]);
    }

    printf("...\nLast 3 elements:\n");
    for (int i = 97; i < 100; i++) {
        printf("Index %d: %d * 2 = %d\n", i, h_in[i], h_out[i]);
    }

    // 8. Clean up VRAM
    cudaFree(d_in); 
    cudaFree(d_out);

    return 0;
}