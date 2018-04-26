// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
	
__kernel
void matrixTranspose(__global float * output,
                     __global float * input,
                     const    uint    width)

{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if(i*width+ j < width* width)
		output[i*width + j] = input [j*width + i];

}


__kernel
void matrixTransposeLocal(__global float * output,
                          __global float * input,
                          __local float * block,
                          const    uint    width)

{
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int blockIdxx = get_group_id(0);
	int blockIdxy = get_group_id(1);
	
	int threadIdxx = get_local_id(0);
	int threadIdxy = get_local_id(1);

	int pos = i*width + j;
	if(pos >= 0 && pos < width*width){
	//DOs casos:
			
		//Que el bloque NO esté en la diagonal de la matriz
		
		if(blockIdxx != blockIdxy)
		{	
			//Trasponemos al copiar
			block[threadIdxx*BLOCK_SIZE + threadIdxy] = input [i*width + j];

			//Copiamos el bloque traspuesto
			output[j*width + i] = block[threadIdxy* BLOCK_SIZE+ threadIdxx];		
		
		}
		//Que el bloque que cogemos esté en la diagonal de la matriz
		else{
			//Trasponemos al copiar
			block[threadIdxx*BLOCK_SIZE + threadIdxy] = input [i*width + j];
			
			//Pegamos en el mismo sitio, ya que estamos en la diagonal
			output[i*width + j] = block[threadIdxy * BLOCK_SIZE + threadIdxx];
			

		}		
	} 	

}
