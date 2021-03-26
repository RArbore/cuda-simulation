#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

__global__
void function (int n, float *x, float *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) *(y + i) = *(x + i) + *(y + i);
}

int main (int argc, char *argv[]) {

    // Create arrays

    int N = 1000000;
    float *x, *y, *d_x, *d_y;

    x = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize host arrays

    for (int i = 0; i < N; i++) {
        *(x + i) = (float) i;
        *(y + i) = (float) i;
    }

    // Move data to GPU

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch function

    int BLOCK_SIZE = 250;
    int BLOCKS = N / BLOCK_SIZE;

    function <<< BLOCKS, BLOCK_SIZE >>> (N, d_x, d_y);

    // Move result to host

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 1000 results

    for (int i = 0; i < 1000; i++) {
        printf("%f\n", *(y + i));
    }

    // Clean-up

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}
