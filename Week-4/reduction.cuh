#ifndef REDUCTIONS_CUH
#define REDUCTIONS_CUH

// Universal signature: 
// n = total elements, d_in = source, d_out = per-block results
typedef void (*reduction_kernel)(int n, const int* d_in, int* d_out);

// --- Version 0: Naive Global Memory ---
__global__ void g_reduce0(int n, const int* d_in, int* d_out);

// --- Version 1: Shared Memory ---
__global__ void g_reduce1(int n, const int* d_in, int* d_out);

// --- Version 2: Warp Shuffle ---
__global__ void g_reduce2(int n, const int* d_in, int* d_out);
// --- Version 3: Grid Stride + Shuffle ---
__global__ void g_reduce3(int n, const int* d_in, int* d_out);

#endif