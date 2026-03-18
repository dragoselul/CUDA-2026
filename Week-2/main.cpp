#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include "sum_reduction.cuh"

enum class DataType { ONES, RANDOM };

void run_benchmark(int n, reduction_func func, std::string methodName, DataType dType) {
    // 1. Initialize Host Data
    std::vector<int> h_in(n);
    int expected_sum = 0;

    if (dType == DataType::ONES) {
        std::fill(h_in.begin(), h_in.end(), 1);
        expected_sum = n;
    } else {
        std::mt19937 rng(1337); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 10);
        for(int &val : h_in) {
            val = dist(rng);
            expected_sum += val;
        }
    }

    // 2. GPU Setup
    int *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    // 3. Timing
    auto start = std::chrono::high_resolution_clock::now();

    // Pass 1: Global reduction to partial sums
    func(blocks, threads, d_in, d_out, n);
    
    // Pass 2: Reduce partial sums to final result (if blocks > 1)
    if (blocks > 1) {
        func(1, threads, d_out, d_out, blocks);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // 4. Verify & Output
    int h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "[" << methodName << "] Size: " << n 
              << " | Time: " << ms.count() << "ms"
              << " | Status: " << (h_out == expected_sum ? "PASS" : "FAIL") 
              << " (Got: " << h_out << ", Expected: " << expected_sum << ")" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
}

int main(int argc, char** argv) {
    int n = 1000000;
    if (argc > 1) n = std::stoi(argv[1]);

    std::cout << "--- Running Benchmarks on GH200 ---" << std::endl;
    
    // You can now easily call different versions with different data
    run_benchmark(n, launch_reduce0, "Reduce0_Ones", DataType::ONES);
    run_benchmark(n, launch_reduce0, "Reduce0_Random", DataType::RANDOM);

    return 0;
}