#include <cuda.h>
#include <math.h>

#include "kernel.h"

#define PI 3.141593
#define BLOCK_SIZE 16 


void /*__global__*/ cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{
	/* To fill */
 	int i = 2;
	int j = 2;

	int begin = i*width*j;
	dim3 threadsPerBlock(8,8);
	dim3 numBlocks(width/threadsPerBlock.x,
			height/threadsPerBlock.y);
	

	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	for (int a = begin; a <= aEnd; a += aStep) {
		
	}
}

__device__
noiseReductionGPU(float *im_in, float * im_NR_out, int h, int w){

	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(i > 2 && i < w-2 && j >2 && j < h-2)
	{
		im_NR_out[i*w+j] = ( 
		im_in[(i-2) * w + j-2] * 2.0 + im_in[(i-2) * w + j-1] * 4.0  + im_in[(i-2) * w + j] * 5.0  + im_in[(i-2) * w + j+1] * 4.0  + im_in[(i-2) * w + j+2] * 2.0 +
		im_in[(i-1) * w + j-2] * 4.0 + im_in[(i-1) * w + j-1] * 9.0  + im_in[(i-1) * w + j] * 12.0 + im_in[(i-1) * w + j+1] * 9.0  + im_in[(i-1) * w + j+2] * 4.0 +
		im_in[(i  ) * w + j-2] * 5.0 + im_in[(i  ) * w + j-1] * 12.0 + im_in[(i  ) * w + j] * 15.0 + im_in[(i  ) * w + j+1] * 12.0 + im_in[(i  ) * w + j+2] * 5.0 +	
		im_in[(i+1) * w + j-2] * 4.0 + im_in[(i+1) * w + j-1] * 9.0  + im_in[(i+1) * w + j] * 12.0 + im_in[(i+1) * w + j+1] * 9.0  + im_in[(i-2) * w + j+2] * 4.0 +
		im_in[(i+2) * w + j-2] * 2.0 + im_in[(i+2) * w + j-1] * 4.0  + im_in[(i+2) * w + j] * 5.0  + im_in[(i+2) * w + j+1] * 4.0  + im_in[(i+2) * w + j+2] * 2.0
		/159.0); 		
	}	

}

__device__ (float *im_in, float * im_GR_out, int h, int w)
imageGradientGPU

