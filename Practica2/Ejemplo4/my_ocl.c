#include "common.c"
#include <stdio.h>
#include "my_ocl.h"
#include <CL/cl.h>

#define BLOCK_SIZE 2


void remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{

	printf("Not Implemented yet!!\n");
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

	cl_mem dInput; //Image input
	cl_mem dOutput; //Image output
	cl_mem localMem;

	
	

//----------------------------------- LECTURA DE KERNEL ----------------------------------

	fp = fopen("removeNoise.cl","r");
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
		//return EXIT_FAILURE;
		return;
	}
	
	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
//		return EXIT_FAILURE;
		return;
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
		//return EXIT_FAILURE;
		return;
	}

	err = output_device_info(device_id);

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		//return EXIT_FAILURE;
		return;
	}	

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		//return EXIT_FAILURE;
		return;
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

	kernel = clCreateKernel(program, "remNoiseKernel", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}

	const int IMG_SIZE = sizeof(float)*height * width;
	
		// create buffer objects to input and output args of kernel function

	dInput = clCreateBuffer(context, CL_MEM_READ_ONLY, IMG_SIZE, NULL, NULL);
	dOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, IMG_SIZE, NULL, NULL);


	err = clEnqueueWriteBuffer(commands, dInput, CL_TRUE, 0, IMG_SIZE, im, 0, NULL, NULL);
    	if (err != CL_SUCCESS)
    	{
        printf("Error: Failed to write dInput with source image!\n%s\n", err_code(err));
        exit(1);
    	}	
    	
	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &dInput) || 		//image in
         clSetKernelArg(kernel, 1, sizeof(cl_mem), &dOutput) ||			//image out
         clSetKernelArg(kernel, 2, (BLOCK_SIZE*BLOCK_SIZE *sizeof(cl_mem)), NULL) || //Local memory
         clSetKernelArg(kernel, 3, sizeof(cl_float), &thredshold) ||		//Thredshold
         clSetKernelArg(kernel, 4, sizeof(cl_int), &window_size) ||		//window size
         clSetKernelArg(kernel, 5, sizeof(cl_int), &height) ||  		//height
         clSetKernelArg(kernel, 6, sizeof(cl_int), &width) != CL_SUCCESS) 	//width
          
         
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= width;
	global[1]= height;

	printf("%d, %d", global[0], global[1]);

	//Como el Dim3 de CUDA, OpenCl necesita un numero en 3 dimensiones para mapear la cantidad de 
	//Hilos y bloques
	
	size_t localDim[2];
	localDim[0] = BLOCK_SIZE;
	localDim[1] = BLOCK_SIZE; 

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
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

	// read the output back to host memory
	err = clEnqueueReadBuffer(commands, dOutput, CL_TRUE, 0, IMG_SIZE, image_out, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%d\n",err);
		exit(1);
	}


//	printMATRIX(array1D, n);
//	printMATRIX(array1D_trans_GPU, n);

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	clReleaseMemObject(dInput);
	clReleaseMemObject(dOutput);



}


