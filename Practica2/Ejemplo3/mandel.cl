
__global typedef struct {unsigned char r, g, b;} rgb_t;


__kernel
void mandelKernel (__global void * text, const int w, const int h,
			 const int scale,const double cx,const double cy){
	
	int i = get_global_id(0);
	int j = get_global_id(1);
	

	double x, y, zx, zy, zx2, zy2;
	int iter;
	int max_iter = 256;
	int min = 256, max = 0;

		
	if(i*h+ j < w* h)
	{	
		rgb_t * pixel;
		
		pixel = &text[(i*w*sizeof(rgb_t))+j];
		
		//rgb_t * pixel = (rgb_t *) (text[i*(w*sizeof(rgb_t))+j]);
		/*
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
			pixel = iter;
		
		//hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);	
		/*
		
		
		if (min == max) max = min + 1;
		if (invert) hue = max - (hue - min);
		if (!saturation) {
			p->r = p->g = p->b = 255 * (max - hue) / (max - min);
			return;
		}
		double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
	#	define VAL 255
		double c = VAL * saturation;
		double X = c * (1 - fabs(fmod(h, 2) - 1));
	 
		p->r = p->g = p->b = 0;
	 
		switch((int)h) {
		case 0: p->r = c; p->g = X; return;
		case 1: p->r = X; p->g = c; return;
		case 2: p->g = c; p->b = X; return;
		case 3: p->g = X; p->b = c; return;
		case 4: p->r = X; p->b = c; return;
		default:p->r = c; p->b = X;
		}	*/
		
	}

}
