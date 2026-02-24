#include <cuda_runtime.h>

__global__ void reduce0_kernel(const int *g_idata, int *g_odata, int n) {
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < static_cast<unsigned int>(n)) ? g_idata[i] : 0;
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
    if ((tid % (2 * s) == 0) && (tid + s < blockDim.x)) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

static int pick_threads(int n) {
  int t = 1;
  while (t < n && t < 1024) {
    t <<= 1;
  }
  return t;
}

void reduce0(const int *d_in, int *d_out, int n) {
  if (n <= 0) {
    return;
  }

  int threads = pick_threads(n);
  int max_blocks = (n + threads - 1) / threads;

  int *d_buf1 = nullptr;
  int *d_buf2 = nullptr;
  cudaMalloc(&d_buf1, max_blocks * sizeof(int));
  cudaMalloc(&d_buf2, max_blocks * sizeof(int));

  const int *d_curr = d_in;
  int *d_tmp = d_buf1;
  int curr_n = n;

  while (true) {
    int blocks = (curr_n + threads - 1) / threads;
    reduce0_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_curr, d_tmp,
                                                               curr_n);
    if (blocks == 1) {
      break;
    }
    curr_n = blocks;
    const int *d_next = d_tmp;
    d_tmp = (d_tmp == d_buf1) ? d_buf2 : d_buf1;
    d_curr = d_next;
  }

  cudaMemcpy(d_out, d_tmp, sizeof(int), cudaMemcpyDeviceToDevice);
  cudaFree(d_buf1);
  cudaFree(d_buf2);
}
