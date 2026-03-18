#ifndef SUM_REDUCTION_CUH
#define SUM_REDUCTION_CUH

typedef void (*reduction_func)(int, int, const int*, int*, int);

#ifdef __cplusplus
extern "C" {
#endif

void launch_reduce0(int blocks, int threads, const int *d_in, int *d_out, int n);
// Add future versions here: launch_reduce1, launch_reduce2, etc.

#ifdef __cplusplus
}
#endif

#endif