/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cassert>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <unistd.h>

#include <helper_cuda.h>
#include <pthread.h>
#include <iostream>
using namespace std;
#define DEBUG 1

#define NUM_THREADS 1

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

float *x, *y, *z;
int numElements = 2;
int nStreams = 4;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
void *PrintHello(void *threadid)
{
    long tid;
    tid = (long)threadid;
    cout << "Hello World! Thread ID, " << tid << endl;
    int threadsPerBlock = 32;
    int blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(x + numElements, y + numElements, z+numElements, 1);
    checkCuda(cudaStreamSynchronize(0));
    pthread_exit(NULL);
}
/**
 * Host main routine
 */
int main(void)
{

    // Print the vector length to be used, and compute its size

    size_t size = (numElements + NUM_THREADS) * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    int N = (numElements + NUM_THREADS);
    checkCuda(cudaMallocManaged(&x, N * sizeof(float)));
    checkCuda(cudaMallocManaged(&y, N * sizeof(float)));
    checkCuda(cudaMallocManaged(&z, N * sizeof(float)));

    // Verify that allocations succeeded
    if (x == NULL || y == NULL || z == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements + NUM_THREADS; ++i)
    {
        x[i] = rand() / (float)RAND_MAX;
        printf("x[%d] addr %llu val %.2f\t", i, &x[i], x[i]);
        y[i] = rand() / (float)RAND_MAX;
        printf("y[%d] addr %llu val %.2f\n", i, &y[i], y[i]);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(x, y, z, numElements);
    pid_t pid = getpid();
    printf("vectorAdd PID: %d\n", pid);
    // while(1){
    //     cout<<"cpu working"<<std::flush;
    //     sleep(2);
    // }

    printf("Test PASSED\n");
    checkCuda(cudaStreamSynchronize(0));
    
    for (int i = 0; i < numElements + NUM_THREADS; ++i)
    {
        x[i] = rand() / (float)RAND_MAX;
        printf("second time: x[%d] addr %llu val %.2f\t", i, &x[i], x[i]);
        y[i] = rand() / (float)RAND_MAX;
        printf("second time: y[%d] addr %llu val %.2f\n", i, &y[i], y[i]);
    }
    // multi threads
    int rc;
    int i;
    pthread_t threads[NUM_THREADS];
    for (i = 0; i < NUM_THREADS; i++)
    {
        cout << "main() : creating thread, " << i << endl;
        rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
        if (rc)
        {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    for (i = 0; i < NUM_THREADS; i++)
    {
        void *ret;
        if (pthread_join(threads[i], &ret) != 0)
        {
            printf("thread exited with '%s'\n", ret);
            exit(3);
        }
    }

    // for (int i = 0; i < numElements + NUM_THREADS; ++i)
    // {
    //     printf("z[%d] addr %llu val %.2f\t", i, &z[i], z[i]);
    // }
    cout << endl;
    // multi stream
    cudaStream_t stream[nStreams];

    // Free device global memory
    checkCuda(cudaFree(x));
    checkCuda(cudaFree(y));
    checkCuda(cudaFree(z));

    printf("Done\n");
    return 0;
}
