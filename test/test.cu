#include<iostream>
#include "cuda_runtime.h"

__global__ void addone(int *a)
{
    *a = *a + 1;
    printf("add one \n");
}


int main()
{
    int a = 0;
    int *d_a;

    cudaMalloc(&d_a, sizeof(int));
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);

    addone<<<1,32>>>(d_a);


    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);


    std::cout<<a<<std::endl;

    cudaFree(d_a);

    return 0;
}