#include <iostream>
#include <math.h>
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += 1)
    x[i] = x[i] + i;
}
 
int main(void)
{
  int N = 16*1024+8*1024;;
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }
  int blockSize = 32;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x);
  for(int j = 0; j < 100000; j++)
   for (int i = 0; i < N; i++) {
    y[i] = 2.0f;
  }
  add<<<numBlocks, blockSize>>>(N, y);


  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
//     maxError = fmax(maxError, fabs(y[i]-3.0f));
//   std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}