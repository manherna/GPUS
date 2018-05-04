
typedef struct {unsigned char r, g, b;} rgb_t;
__kernel
void mandelKernel (__global rgb_t ** texture,__global rgb_t * pixel, const int w, const int h,
			 const int scale,const double cx,const double cy)

	int i = get_global_id(0);
	int j = get_global_id(1);

	double x, y, zx, zy, zx2, zy2;
	int iter;
	int max_iter = 256;
	int min = 256, max = 0;
		
	if(i*h+ j < w* h)
	{
		pixel = tex[j];
		
		y = (j - h/2) * scale + cy;		
		x = (i - w/2) * scale + cx;
		
		zx = zy = zx2 = zy2 = 0;
		
		 for (iter=0; iter < max_iter; iter++) {
				zy=2*zx*zy + y;
				zx=zx2-zy2 + x;
				zx2=zx*zx;
				zy2=zy*zy;
				if (zx2+zy2>max_iter)
					break;
			}
			if (iter < min) min = iter;
			if (iter > max) max = iter;
			*(unsigned short *)pixel = iter;
		
		/*esto habrá que hacer otro método a parte igual que está en el interactive*/
		
		//hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);	
	}

}
