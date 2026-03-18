#include <stdio.h>

__global__ void copy (float *odata, float *idata, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
        odata[idx] = idata[idx];
}


int main() {
    return 0;
}