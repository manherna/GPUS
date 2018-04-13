#include <cuda.h>
#include <math.h>

#include "kernel.h"

#define PI 3.141593
#define BLOCK_SIZE 16


__global__
void noiseReductionGPU(float *im, float * im_NR_out, int h, int width){

	uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
	uint x = threadIdx.x;
	uint y = threadIdx.y;

	

	__shared__ float shared_block [BLOCK_SIZE+4][BLOCK_SIZE+4];
	 

	shared_block[x+2][y+2] = im[i*width+j];

	//LLenado de la memoria compartida
	//Llenamos las de arriba
	//SI está en la primera fila del bloque Y no es la primera fila de la foto
	if(threadIdx.y == 0 && i > 2) {
		shared_block[threadIdx.x+2][threadIdx.y+1] = im[(i-1)*width+j];
		shared_block[threadIdx.x+2][threadIdx.y  ] = im[(i-2)*width+j];
		//LLenamos la esquina superior izquierda del bloque
		if(threadIdx.x == 0 && j > 2){
			shared_block[threadIdx.x  ][threadIdx.y  ] = im[(i-2)*width+(j-2)];
			shared_block[threadIdx.x  ][threadIdx.y+1] = im[(i-1)*width+(j-2)];
			shared_block[threadIdx.x+1][threadIdx.y  ] = im[(i-2)*width+(j-1)];
			shared_block[threadIdx.x+1][threadIdx.y+1] = im[(i-1)*width+(j-1)];
		}
		
	}
	//Llenamos las de abajo
	if(threadIdx.y == BLOCK_SIZE-1 && i < h-2){
		shared_block[threadIdx.x+2][threadIdx.y+3] = im[(i+1)*width+j];
		shared_block[threadIdx.x+2][threadIdx.y+4]   = im[(i+2)*width+j];
		//Llenamos la esquina inferior derecha del bloque
		if(threadIdx.x == BLOCK_SIZE-1 && j < width-2){
			shared_block[threadIdx.x+3][threadIdx.y+3] = im [(i+1)*width+(j+1)];
			shared_block[threadIdx.x+3][threadIdx.y+4] = im [(i+2)*width+(j+1)];
			shared_block[threadIdx.x+4][threadIdx.y+3] = im [(i+1)*width+(j+2)];
			shared_block[threadIdx.x+4][threadIdx.y+4] = im [(i+2)*width+(j+2)];		
		}

	}
	//Lateral Izquierdo
	//Si es la primera columna del bloque Y no es la primera columna de la foto, esto es (j == 0)
	if(threadIdx.x == 0 && j > 2 ){
		shared_block[threadIdx.x+1][threadIdx.y+2] = im [i * width+ (j-1)];
		shared_block[threadIdx.x  ][threadIdx.y+2] = im [i * width+ (j-2)];
		//LLenamos la esquina inferior izquierda del bloque
		if(threadIdx.y == BLOCK_SIZE -1 && i < h-2){
			shared_block[threadIdx.x  ][threadIdx.y+4] = im [(i+2)*width+(j-2)];
			shared_block[threadIdx.x  ][threadIdx.y+3] = im [(i+1)*width+(j-2)];
			shared_block[threadIdx.x+1][threadIdx.y+4] = im [(i+2)*width+(j-1)];
			shared_block[threadIdx.x+1][threadIdx.y+3] = im [(i+1)*width+(j-1)];
		}
	
	}
	//Lateral derecho
	if(threadIdx.x == BLOCK_SIZE-1 && j < width-2){
		shared_block[threadIdx.x+3][threadIdx.y+2] = im [i*width +(j+1)];
		shared_block[threadIdx.x+4][threadIdx.y+2] = im [i*width +(j+2)];
		//Llenamos la esquina superior derecha del bloque
		if(threadIdx.y == 0 && i > 2){
			shared_block[threadIdx.x+3][threadIdx.y+1] = im[(i-1)*width + (j+1)];
			shared_block[threadIdx.x+4][threadIdx.y+1] = im[(i-1)*width + (j+2)];
			shared_block[threadIdx.x+3][threadIdx.y  ] = im[(i-2)*width + (j+1)];
			shared_block[threadIdx.x+4][threadIdx.y  ] = im[(i-2)*width + (j+2)];	
		}

	}
	
	__syncthreads();
	
	if(i > 2 && i < h-2 && j >2 && j < width-2)
	{
		
		im_NR_out[i*width+j] = 
				 (2.0*shared_block[x  ][y  ] +  4.0*shared_block[x+1][y  ] +  5.0*shared_block[x+2][y  ] +  4.0*shared_block[x+3][y  ] + 2.0*shared_block[x+4][y  ]
				+ 4.0*shared_block[x  ][y+1] +  9.0*shared_block[x+1][y+1] + 12.0*shared_block[x+2][y+1] +  9.0*shared_block[x+3][y+1] + 4.0*shared_block[x+4][y+1] 
				+ 5.0*shared_block[x  ][y+2] + 12.0*shared_block[x+1][y+2] + 15.0*shared_block[x+2][y+2] + 12.0*shared_block[x+3][y+2] + 5.0*shared_block[x+4][y+2]
				+ 4.0*shared_block[x  ][y+3] +  9.0*shared_block[x+1][y+3] + 12.0*shared_block[x+2][y+3] +  9.0*shared_block[x+3][y+3] + 4.0*shared_block[x+4][y+3]
				+ 2.0*shared_block[x  ][y+4] +  4.0*shared_block[x+1][y+4] +  5.0*shared_block[x+2][y+4] +  4.0*shared_block[x+3][y+4] + 2.0*shared_block[x+4][y+4])
				/159.0;		
	}
	
	else if (i<h && j<width) im_NR_out[i*width+j] = 0.0;	

}

__device__ 
void imageGradientGPU (float *im_in, float * im_GR_out, int h, int w) {

}


void /*__global__*/ cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	
	dim3 numBlocks(ceil(width/(float)threadsPerBlock.x),ceil(
			height/(float)threadsPerBlock.y));

	//Variables para guardar la imagen antes y después
	float * im_GPU;
	float * im_out_GPU;
	int IMG_SIZE = height*width*sizeof(float);

	//Reserva de memoria
	cudaMalloc((void**)&im_GPU, IMG_SIZE);
	cudaMalloc((void**)&im_out_GPU, IMG_SIZE);
	cudaMemcpy(im_GPU,im, IMG_SIZE, cudaMemcpyHostToDevice);

	//LLamada a las funciones de tratado de imagen
	noiseReductionGPU<<<numBlocks,threadsPerBlock>>>(im_GPU, im_out_GPU, height, width);
	

	//Devolución de valores y liberado de memoria.
	cudaMemcpy(image_out, im_out_GPU, IMG_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(im_GPU);	
	cudaFree(im_out_GPU);	

}
