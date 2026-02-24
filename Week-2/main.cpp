#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "sum_reduction.cuh"

int main() {
	const int n = 1'000'000;
	std::vector<int> h_in(n, 1);

	int *d_in = nullptr;
	int *d_out = nullptr;
	cudaMalloc(&d_in, n * sizeof(int));
	cudaMalloc(&d_out, sizeof(int));
	cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	reduce0(d_in, d_out, n);
	int result = 0;
	cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "sum: " << result << "\n";

	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
