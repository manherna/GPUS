#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printGrid(float * G, int fils, int cols){
for(int i = 0; i < fils; i++) {
	for(int j = 0; j<cols; j++) printf("%f ", G[i*fils+j]);	
		printf("\n");
	}
}


__global__ 
void jacobi2d(float * U, float * U_new , float * F, int n, int m){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	
	if(i > 0 && i< (m-1) && j > 0 && j< (n-1)){
		U_new[i*n+j] = U[(i-1)*n+j] + U[(i+1)*n+j] + U[i*n+(j-1)] + U[i*n+(j+1)] -F[i*n+j];
	}

}



int main(int argc, char *argv[])
{
	srand(time(NULL));
	int n = atoi (argv [1]);
	int m = atoi (argv [2]);
	
	int size = n * m * sizeof(float);
	
	float * U_d, * N_d;
	float * U, * N, * F;		
	
	U = (float *)malloc(size);
	N = (float *)malloc(size);
	F = (float *)malloc (size);


	for(int i = 0; i < n*m; i++){
		F[i] = 1;
		U[i] = (float)(rand() % 450 + 1);
	}
	printGrid(F, m, n);
	printf("--------------\n");	
	printGrid(U, m, n);
	printf("--------------\n");


	cudaMalloc((void **) &U_d, size);
	cudaMemcpy(U_d, U, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &N_d, size);

	dim3 dimBloque (256, 256, 1);
	dim3 numHilos (n/256.0, 1, 1);


	jacobi2d<<<numHilos,dimBloque>>>(U_d, N_d, F, n, m);
	

	cudaMemcpy(N, N_d, size, cudaMemcpyDeviceToHost);


	printGrid(N, n, m);
	cudaFree(N_d);
	cudaFree(U_d);
	free(U);
	free(N);
	


	
}
