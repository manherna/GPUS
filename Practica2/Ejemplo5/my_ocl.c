#include "common.c"
#include <stdio.h>
#include "my_ocl.h"
#include <CL/cl.h>


double calc_piOCL(int n)
{
	
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
	
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	
	
	//----------------------------------- LECTURA DE KERNEL ----------------------------------

	fp = fopen("pi_kernel.cl","r");
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

	kernel = clCreateKernel(program, "calc_Pi_kernel", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}
	
	//Declaración de objetos de memoria
	double * piValues = (double *)malloc(sizeof(double)*n);
	cl_mem piOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*n, NULL, NULL);
	
		
    	
	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_int), &n) ||
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &piOutput) != CL_SUCCESS)		
     
         
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}
	
	
	const size_t global = n;
	const size_t local = 2;
	
	double t0d = getMicroSeconds();
	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                               &global, &local, 0, NULL, NULL);
                               
             
	double t1d = getMicroSeconds();

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);
	
	
	// the output back to host memory
	err = clEnqueueReadBuffer(commands, piOutput, CL_TRUE, 0, sizeof(double)*n, piValues, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%d\n",err);
		exit(1);
	}
	
	double acc = 0.0;
	int x;
	for(x = 0; x < n; x++)
		acc +=  piValues[x];
		
	double pi = acc /n;
	
	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	free(piValues);
	clReleaseMemObject(piOutput);

	return pi;
}


