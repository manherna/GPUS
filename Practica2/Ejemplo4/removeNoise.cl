#define MAX_WINDOW_SIZE 5*5

__kernel 
void remNoiseKernel(__global float * im, __global float * im_out, __local float * locMem, float thredshold, int win_size, int h, int w)
{
	int thIdx = get_local_id(0);
	int thIdy = get_local_id(1);
	
	int globalPx = get_global_id(0);
	int globalPy = get_global_id(1);
	

	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (win_size-1)>>1; 

	im_out[globalPy * w + globalPx] = 0.0;
	
	if(globalPx >=ws2 && globalPx < w-ws2 && globalPy >= ws2 && globalPy < h-ws2)
	{
		//im_out[globalPy * w + globalPx] = im[globalPy * w + globalPx];		
		
		for (int ii =-ws2; ii<=ws2; ii++)
				for (int jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*win_size + jj+ws2] = im[(globalPy+ii)*w + globalPx+jj];
					
		
	int i, j;
	float tmp;
	
	int size = win_size * win_size;
	for (i=1; i<size; i++){
		for (j=0 ; j<size - i; j++)
			if (window[j] > window[j+1]){
				tmp = window[j];
				window[j] = window[j+1];
				window[j+1] = tmp;
			}
	}
	
	median = window[(win_size*win_size-1)>>1];
	if (fabs((median-im[globalPy*w+globalPx])/median) <=thredshold)
		im_out[globalPy*w+ globalPx] = im[globalPy*w+globalPx];
	else
		im_out[globalPy*w+ globalPx] = median;

	}

}
	
