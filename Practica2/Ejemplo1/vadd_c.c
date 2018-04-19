//------------------------------------------------------------------------------
//
// Name:       vadd.c
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Modified by Carlos Garcia, April 2014
//             
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


/* From common.c */
extern double getMicroSeconds();
extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);
extern float *getmemory1D( int nx );

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd  
//
// Purpose: Compute the elementwise sum c = a+b
// 
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//
 
const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    int          err;               // error code returned from OpenCL calls
    float        *h_a;              // a vector 
    float        *h_b;              // b vector 
    float        *h_c;              // c vector (a+b) returned from the compute device
    unsigned int correct;           // number of correct results  

    size_t global;                  // global domain size  

    cl_device_id     device_id;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel
    
    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the output c vector
    
    int i;
    int length;
    if (argc==2)
        length = atoi(argv[1]);
    else {
        length = 1024;
        printf("./exec length (by default length=%i)\n", length);
    }

    // Fill vectors a and b with random float values
    h_a = getmemory1D( length );
    h_b = getmemory1D( length );
    h_c = getmemory1D( length );
    init1Drand(h_a, length);
    init1Drand(h_b, length);
    
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
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program 
    ko_vadd = clCreateKernel(program, "vadd", &err);
    if (!ko_vadd || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Create the input (a, b) and output (c) arrays in device memory
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * length, NULL, NULL);
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * length, NULL, NULL);
    d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * length, NULL, NULL);
    if (!d_a || !d_b || !d_c)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write a and b vectors into compute device memory 
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * length, h_a, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write h_a to source array!\n%s\n", err_code(err));
        exit(1);
    }

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * length, h_b, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write h_b to source array!\n%s\n", err_code(err));
        exit(1);
    }
	
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &length);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments!\n");
        exit(1);
    }

    double t0 = getMicroSeconds();
	
    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = length;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }

    // Wait for the commands to complete before stopping the timer
    clFinish(commands);

    double t1 = getMicroSeconds();
    printf("\nThe kernel ran in %lf seconds\n",(t1-t0)/1000000);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * length, h_c, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }
    
    // Test the results
    correct = 0;
    float tmp;
    
    for(i = 0; i < length; i++)
    {
        tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];             // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
        }
    }
    
    // summarize results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, length);
    
    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

