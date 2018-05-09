#define MAX_WINDOW_SIZE 5*5

__kernel 
void remNoiseKernel(__global float * im, __global float * im_out,__local float * locMem,  float thredshold, int win_size, int h, int w)
{
	int thIdx = get_local_size(0);
	int thIdy = get_local_size(1);
	
	int globalPx = get_global_size(0);
	int globalPy = get_global_size(1);
	
	
	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (win_size-1)>>1; 
	im_out[globalPy * w + globalPx] = im[globalPy * w + globalPx];
	
	if(globalPx >= ws2 && globalPx < w-ws2 && globalPy >= ws2 && globalPy < h-ws2){
		
		
	
	
	}
	
/*
	for(i=ws2; i<height-ws2; i++)
		for(j=ws2; j<width-ws2; j++)
		{
			for (ii =-ws2; ii<=ws2; ii++)
				for (jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*window_size + jj+ws2] = im[(i+ii)*width + j+jj];

			// SORT
			buble_sort(window, window_size*window_size);
			median = window[(window_size*window_size-1)>>1];

			if (fabsf((median-im[i*width+j])/median) <=thredshold)
				image_out[i*width + j] = im[i*width+j];
			else
				image_out[i*width + j] = median;

				
		}*/
}
