#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "common.c"
#include <stdio.h>

#include <CL/cl.h>

#define RUN_SERIAL     0
#define RUN_OPENCL_CPU 1
#define RUN_OPENCL_GPU 2
int run_mode;
 
void set_texture();
 
typedef struct {unsigned char r, g, b;} rgb_t;
rgb_t **tex = 0;
int gwin;
GLuint texture;
int width, height;
int tex_w, tex_h;
double scale = 1./256;
double cx = -.6, cy = 0;
int color_rotate = 0;
int saturation = 1;
int invert = 0;
int max_iter = 256;
 
/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;

void render()
{
	double	x = (double)width /tex_w,
			y = (double)height/tex_h;
 
	glClear(GL_COLOR_BUFFER_BIT);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
 
	glBindTexture(GL_TEXTURE_2D, texture);
 
	glBegin(GL_QUADS);
 
	glTexCoord2f(0, 0); glVertex2i(0, 0);
	glTexCoord2f(x, 0); glVertex2i(width, 0);
	glTexCoord2f(x, y); glVertex2i(width, height);
	glTexCoord2f(0, y); glVertex2i(0, height);
 
	glEnd();
 
	glFlush();
	glFinish();
}
 
int shots = 1;
void screen_shot()
{
	char fn[100];
	int i;
	sprintf(fn, "screen%03d.ppm", shots++);
	FILE *fp = fopen(fn, "w");
	fprintf(fp, "P6\n%d %d\n255\n", width, height);
	for (i = height - 1; i >= 0; i--)
		fwrite(tex[i], 1, width * 3, fp);
	fclose(fp);
	printf("%s written\n", fn);
}
 
void keypress(unsigned char key, int x, int y)
{
	switch(key) {
	case 'q':	glFinish();
			glutDestroyWindow(gwin);
			return;
	case 27:	scale = 1./256; cx = -.6; cy = 0; break;
 
	case 'r':	color_rotate = (color_rotate + 1) % 6;
			break;
 
	case '>': case '.':
			max_iter += 64;
			if (max_iter > 1 << 15) max_iter = 1 << 15;
			printf("max iter: %d\n", max_iter);
			break;
 
	case '<': case ',':
			max_iter -= 64;
			if (max_iter < 64) max_iter = 64;
			printf("max iter: %d\n", max_iter);
			break;
 
	case 'm':	saturation = 1 - saturation;
			break;
 
	case 'i':	screen_shot(); return;
	case 'z':	max_iter = 4096; break;
	case 'x':	max_iter = 128; break;
	case 's':	run_mode = RUN_SERIAL; break;
	case 'c':	run_mode = RUN_OPENCL_CPU; break;
	case 'g':	run_mode = RUN_OPENCL_GPU; break;
	case ' ':	invert = !invert;
	}
	set_texture();
}
 
void hsv_to_rgb(int hue, int min, int max, rgb_t *p)
{
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
	}
}

double calc_mandel_opencl()
{

	// OpenCL host variables
	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[2];
	
	
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	cl_mem dImage; //Device input



	// read the kernel
	fp = fopen("mandel.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';

	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Secure a GPU
	int i;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
		{
			break;
		}
	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	err = output_device_info(device_id);

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// create command queue 
	command_queue = clCreateCommandQueue(context,device_id, 0, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create command queue. Error Code=%d\n",err);
		exit(1);
	}
	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create program object. Error Code=%d\n",err);
		exit(1);
	}       
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        	printf("Build failed. Error Code=%d\n", err);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "mandelKernel", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel ob[ject. Error Code=%d\n",err);
		exit(1);
	}

	// create buffer objects to input and output args of kernel function

	int size = width * height * sizeof (float) * sizeof (rgb_t);

	dImage = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

	
	err = clEnqueueWriteBuffer(commands, dImage, CL_TRUE, 0, size, ((void*)tex), 0, NULL, NULL);
    	if (err != CL_SUCCESS)
    	{
        printf("Error: Failed to write h_a to source array!\n%s\n", err_code(err));
        exit(1);
    	}	

	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &dImage) ||
         clSetKernelArg(kernel, 1, sizeof(cl_uint), &width) ||
	 //clSetKernelArg(kernel, 2, (BLOCK_SIZE*BLOCK_SIZE *sizeof(cl_mem)), NULL) || //local memory
         clSetKernelArg(kernel, 2, sizeof(cl_uint), &height) ||
         clSetKernelArg(kernel, 3, sizeof(cl_int), &scale)||
         clSetKernelArg(kernel, 4, sizeof(cl_int), &cx)||
         clSetKernelArg(kernel, 5, sizeof(cl_int), &cy) != CL_SUCCESS)   
         
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= width;
	global[1]= height;

	//Como el Dim3 de CUDA, OpenCl necesita un numero en 3 dimensiones para mapear la cantidad de 
	//Hilos y bloques
	
	size_t localDim[2];
	localDim[0] = 2;
	localDim[1] = 2;

	// Enqueue the kernel object with 
	// Dimension size = 2, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	double t0d = getMicroSeconds();
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                   global, localDim, 0, NULL, NULL);
	double t1d = getMicroSeconds();

	if (err != CL_SUCCESS)
	{	unsigned short *)
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

	// read the output back to host memory
	//err = clEnqueueReadBuffer(commands, dOutput, CL_TRUE, 0, size, array1D_trans_GPU, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%d\n",err);
		exit(1);
	}



	
	
		


//	printMATRIX(array1D, n);
//	printMATRIX(array1D_trans_GPU, n);

	// clean upunsigned short *)
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);

	return 0;
}
 
double calc_mandel()
{
	int i, j, iter, min, max;
	rgb_t *pixel;
	double x, y, zx, zy, zx2, zy2;
	double t0;

	t0 = getMicroSeconds();
	min = max_iter; max = 0;
	for (i = 0; i < height; i++) {
		pixel = tex[i];
		y = (i - height/2) * 
		scale + cy;
		for (j = 0; j  < width; j++, pixel++) {
			x = (j - width/2) * scale + cx;

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
		}
	}
 
	for (i = 0; i < height; i++)
		for (j = 0, pixel = tex[i]; j  < width; j++, pixel++)
			hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);

	return(getMicroSeconds()-t0);
}
 
void alloc_tex()
{
	int i, ow = tex_w, oh = tex_h;
 
	for (tex_w = 1; tex_w < width;  tex_w <<= 1);
	for (tex_h = 1; tex_h < height; tex_h <<= 1);
 
	if (tex_h != oh || tex_w != ow)
		tex = realloc(tex, tex_h * tex_w * 3 + tex_h * sizeof(rgb_t*));
 
	for (tex[0] = (rgb_t *)(tex + tex_h), i = 1; i < tex_h; i++)
		tex[i] = tex[i - 1] + tex_w;
}
 
void set_texture()
{
	double t;
	char title[128];

	alloc_tex();
	switch (run_mode){
		case RUN_SERIAL:	   t=calc_mandel(); break;
		case RUN_OPENCL_CPU: t=calc_mandel_opencl(); break;
		case RUN_OPENCL_GPU: t=calc_mandel_opencl();
	};

 
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_w, tex_h,
		0, GL_RGB, GL_UNSIGNED_BYTE, tex[0]);
 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	render();

	sprintf(title, "Mandelbrot: %5.2f fps (%ix%i)", 1000000/t, width, height);
	glutSetWindowTitle(title);
}
 
void mouseclick(int button, int state, int x, int y)
{
	if (state != GLUT_UP) return;
 
	cx += (x - width / 2) * scale;
	cy -= (y - height/ 2) * scale;
 
	switch(button) {
	case GLUT_LEFT_BUTTON: /* zoom in */
		if (scale > fabs(x) * 1e-16 && scale > fabs(y) * 1e-16)
			scale /= 2;
		break;
	case GLUT_RIGHT_BUTTON: /* zoom out */
		scale *= 2;
		break;
	/* any other button recenters */
	}
	set_texture();
}
 
 
void resize(int w, int h)
{
	//printf("resize %d %d\n", w, h);
	width = w;
	height = h;
 
	glViewport(0, 0, w, h);
	glOrtho(0, w, 0, h, -1, 1);
 
	set_texture();
}
 
void init_gfx(int *c, char **v)
{
	glutInit(c, v);
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(640, 480);
	glutDisplayFunc(render);
 
	gwin = glutCreateWindow("Mandelbrot");
 
	glutKeyboardFunc(keypress);
	glutMouseFunc(mouseclick);
	glutReshapeFunc(resize);
	glGenTextures(1, &texture);
	set_texture();
}
 
int main(int c, char **v)
{
	init_gfx(&c, v);
	printf("keys:\n\tr: color rotation\n\tm: monochrome\n\ti: screen shot\n\t"
            "s: serial code\n\tc: OpenCL CPU\n\tg: OpenCL GPU\n\t"
		"<, >: decrease/increase max iteration\n\tq: quit\n\tmouse buttons to zoom\n");
 
	glutMainLoop();
	return 0;
}
