#include <stdio.h>


// Compute vector sum C = A+B
__global__
void vecAddkernel(float* A_d, float* B_d, float* C_d, int n)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if(i<n) C_d[i] = A_d[i] + B_d[i];

}

int main(int argc, char *argv[])
{

int n = atoi(argv[1]);
float * A, *B, *C;
float* A_d, *B_d, *C_d;
int size = n* sizeof(float);

// A, B and C malloc and init
A = (float*)malloc(size);
B = (float*)malloc(size);
C = (float*)malloc(size);


for(int i = 0; i < n; i++){

	A[i] = (float)i;
	B[i] = (float)i;
	
}

// Get device memory for A, B, C
// copy A and B to device memory
cudaMalloc((void **) &A_d, size);
cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
cudaMalloc((void **) &B_d, size);
cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
cudaMalloc((void **) &C_d, size);


// Kernel execution in device
// (vector add in device)

dim3 DimBlock(256, 1, 1);
dim3 DimGrid(ceil(n/256.0), 1, 1);
vecAddkernel<<<DimGrid,DimBlock>>>(A_d, B_d, C_d, n);
// copy C from device memory
cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);


//present changes
for(int i = 0; i < n; i++)printf("%f ",C[i]);


//free A_d, B_d, C_d
cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);

// free A, B, C
free(A);
free(B);
free(C);


 }
