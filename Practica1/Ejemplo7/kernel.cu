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

	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	for (int a = begin; a <= aEnd; a += aStep) {
		
	}
}
