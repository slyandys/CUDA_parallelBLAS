//System header
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA header
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void CUParaSgemv(const float *a, float *b, float *c,unsigned int size)//valid
{
	unsigned int id =  blockIdx.x * blockDim.x + threadIdx.x;
	//int i =  threadIdx.x;

	if(size<=id)
		return;
	float temp = 0.0;

	for(unsigned int k = 0;k<size; k++)
	{
		if(id < size)
			//Column access - coalesced access
			temp += a[k*size+id] * b[k];

			//Row access 
			//temp += a[id*size+k] * b[k];
	}

	c[id] += temp;
}

//__global__ void transpose(float *odata, float *idata, int width, int height)
//{
//	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
//	
//	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
//	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
//	if((xIndex < width) && (yIndex < height))
//	{
//		unsigned int index_in = yIndex * width + xIndex;
//		block[threadIdx.y][threadIdx.x] = idata[index_in];
//	}
//
//	__syncthreads();
//
//	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
//	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
//	if((xIndex < height) && (yIndex < width))
//	{
//		unsigned int index_out = yIndex * height + xIndex;
//		odata[index_out] = block[threadIdx.x][threadIdx.y];
//	}
//
//	float temp = 0.0;
//	unsigned int idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
//
//	if(idx<height){
//		for(int i=0;i<width;i++)
//		{
//			temp = idata[idx*width+i];
//			odata[i*]
//		}
//	}
//
//}
//void easyTranspose(float o_a[],float i_a[],int size)
//{
//	int col = size*size;
//	for(int i = 0;i<col;i++)
//	{
//		for(int j=0;j<col;j++)
//			o_a[j*col+i]=i_a[i*col+j];
//	}
//}


void simple_sgemv(float *A, float *B, float *C,unsigned int size) //valid
{
	unsigned int i,j;

	for(i = 0;i < size; i++)
	{
		float prod = 0;

		for(j = 0;j < size; j++)
		{	
			prod += A[i * size + j] * B[j];
		}
		C[i] = prod;
	}
}

int main()
{
	//# of nodes(equations)
	//each node has 3-direction displacement
	unsigned int Nodes = 100;						  //threashold 3500-old/4500-new
	unsigned int ARRAY_SIZE = 3*Nodes;				  //Vector Scale;
	unsigned int ARRAY_SIZE2 = ARRAY_SIZE*ARRAY_SIZE; //Matrix Scale;

	//CPU timing
	clock_t start, finish;				//CPU_sgemv time elapse
	clock_t malloc_start,malloc_fin;	//CPU malloc time
	clock_t init_start,init_fin;		//CPU inital time
	clock_t trans_start,trans_fin;		//CPU time on transpose Matrix
	float duration;
	float malloc_duration;
	float init_duration;
	float trans_duration;


	//GPU timing
	float cal_time;
	float cudamalloctime;
	float cudamemcpytime;
	float cudamemcpyout;
	cudaEvent_t d_start, d_stop;
	cudaEvent_t cuda_mallocstart,cuda_mallocfin;
	cudaEvent_t cuda_memcpystart,cuda_memcpyfin;
	cudaEvent_t cuda_memcpyout_start,cuda_memcpyout_fin;


	
	//Host
	float *h_a;
	float *h_b;
	float *h_c;
	float *h_cpu;
	float *h_check;
	float *h_atr;

	//Device
	float *d_a;
	//float *d_atr;
	float *d_b;
	float *d_c;

	//cuda status record
	cudaError_t cudaStatus;

	printf("The nodes number is: %d\n",Nodes);
	printf("The total equations number is : %d\n",ARRAY_SIZE);
	printf("Total bytes will be transfered\n");


	printf("\tMatrix A: %d MB\n",ARRAY_SIZE2*4/1000000);
	printf("\tVector b: %d KB\n",ARRAY_SIZE*4/1000);

	printf("Pre-processing in CPU...\n");
/******Malloc on CPU*******/
	//start the clock
	malloc_start = clock();

	//generate the input array on the host
	h_a=(float*)malloc(sizeof(float)*ARRAY_SIZE2);
    h_b=(float*)malloc(sizeof(float)*ARRAY_SIZE);
    h_c=(float*)malloc(sizeof(float)*ARRAY_SIZE);
	h_cpu=(float*)malloc(sizeof(float)*ARRAY_SIZE);
	h_atr = (float*)malloc(sizeof(float)*ARRAY_SIZE2);
	//h_check=(float*)malloc(sizeof(float)*ARRAY_SIZE2);
	
	//finish time
	malloc_fin = clock();
	//Processing Time in CPU
	malloc_duration = (float)(malloc_fin - malloc_start) / CLOCKS_PER_SEC;
	printf( "\n%f seconds passed in mallocation\n", malloc_duration);
	printf("\n");

/****************************/

/******Initalization on CPU*******/

//use h_ = float(i) to standard the value for the initialization

	init_start = clock();
	
	srand((int)time(0));
	//inital the h_a, h_b
	for(unsigned int i = 0;i<ARRAY_SIZE2;i++){
		h_a[i] = float(i);//rand();//float(i);//rand();
	}
	for(unsigned int i = 0;i<ARRAY_SIZE;i++){
		h_b[i] = float(i);//rand();//float(i);//rand();
	}
	for(unsigned int i = 0;i<ARRAY_SIZE;i++){
		h_c[i] = float(0);
	}
	for(unsigned int i = 0;i<ARRAY_SIZE;i++){
		h_cpu[i] = float(0);
	}

	//time on transpose
	trans_start = clock();
	
	for(unsigned int i = 0;i<ARRAY_SIZE;i++){
		//h_atr[i] = float(0);
		for(unsigned int j=0;j<ARRAY_SIZE;j++)
			h_atr[j*ARRAY_SIZE+i]=h_a[i*ARRAY_SIZE+j];
	}

	trans_fin = clock();
	trans_duration = (float)(trans_fin - trans_start) / CLOCKS_PER_SEC;
	printf( "\n%f seconds passed in transpose..\n", trans_duration);
	
	init_fin = clock();

	//Processing Time on CPU
	init_duration = (float)(init_fin - init_start) / CLOCKS_PER_SEC;
	printf( "\n%f seconds passed in initalizaton\n", init_duration);
	printf("\n");
	printf("******************End Pre-processing.**************\n");

/**********************************/


/**************CPU sgemv calculation time********************/

	start = clock();

	//kernel function on CPU
	simple_sgemv(h_a,h_b,h_cpu,ARRAY_SIZE);

	finish = clock();
	
	//Processing Time in CPU
	duration = (float)(finish - start) ;// CLOCKS_PER_SEC;
	printf( "\n%f milliseconds passed in CPU_sgemv\n", duration);
	printf("\n");

/**********************************/

	//system("pause");
	////Print Result
	//printf("\nThe result Matrix C is:\n");
	//for(unsigned int i=0;i<ARRAY_SIZE;i++){
	//	printf("%f\n", h_cpu[i]);
	//}
	printf("Pre-processing in GPU...\n");
/**************GPU malloc********************/
	cudaEventCreate(&cuda_mallocstart);
	cudaEventCreate(&cuda_mallocfin);
	cudaEventRecord(cuda_mallocstart,0); //mark event
	
	//allocate GPU memory
	//Malloc the memory for matrix and check
	cudaStatus = cudaMalloc((void**)&d_a, sizeof(float)*ARRAY_SIZE2);
    
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMalloc Matrix):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}

	//Malloc the memory for vector and check
	cudaStatus = cudaMalloc((void**)&d_b, sizeof(float)*ARRAY_SIZE);
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMalloc Vector):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}

	//Malloc the memory for storing result and check
    cudaStatus = cudaMalloc((void**)&d_c, sizeof(float)*ARRAY_SIZE);
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMalloc result):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}
	

	cudaThreadSynchronize();
	cudaEventRecord(cuda_mallocfin,0);

	cudaEventSynchronize(cuda_mallocfin);
	cudaEventElapsedTime(&cudamalloctime,cuda_mallocstart,cuda_mallocfin);
	printf( "\n%f milliseconds passed in GPU malloc\n", cudamalloctime );
/*********************************************/
	
/**************GPU Memcpy time********************/
	//Timer
	cudaEventCreate(&cuda_memcpystart);
	cudaEventCreate(&cuda_memcpyfin);
	cudaEventRecord(cuda_memcpystart,0); //mark event

	//transfer the array from Host to device(CPU->GPU) and check the cudaStatus
	
	//Column access
	cudaStatus = cudaMemcpy(d_a, h_atr, sizeof(float)*ARRAY_SIZE2, cudaMemcpyHostToDevice);
	
	//Row access
	//cudaStatus = cudaMemcpy(d_a, h_a, sizeof(float)*ARRAY_SIZE2, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMemcpy matrix):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}
    cudaStatus = cudaMemcpy(d_b, h_b, sizeof(float)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMemcpy vector):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}
	cudaStatus = cudaMemcpy(d_c, h_c, sizeof(float)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMemcpy result):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}

	cudaThreadSynchronize();
	cudaEventRecord(cuda_memcpyfin,0);

	cudaEventSynchronize(cuda_memcpyfin);
	cudaEventElapsedTime(&cudamemcpytime,cuda_memcpystart,cuda_memcpyfin);
	printf( "\n%f milliseconds passed in cuda memory copy\n", cudamemcpytime );

/*********************************************/
	printf("*****************End Pre-processing in GPU********************\n");
/**************GPU Caculation time********************/

	printf("\n*****************A transpose before the calculation********************\n");

	//easyTranspose(h_atr,h_a,ARRAY_SIZE);


	////A transpose Before the calculation...
	//cudaStatus = cudaMalloc((void**)&d_atr, sizeof(float)*ARRAY_SIZE2);

	////Get malloc error
	//cudaStatus = cudaGetLastError();
	//if(cudaStatus != cudaSuccess)
	//{
	//	printf("\nCuda Error(cudaMalloc Matrix):%s\n",cudaGetErrorString(cudaStatus));
	//	system("pause\n");
	//	return 0;
	//}

	////Memory copy
	//cudaStatus = cudaMemcpy(d_atr, h_a, sizeof(float)*ARRAY_SIZE2, cudaMemcpyHostToDevice);
	//if(cudaStatus != cudaSuccess)
	//{
	//	printf("\nCuda Error(cudaMemcpy transpose matrix):%s\n",cudaGetErrorString(cudaStatus));
	//	system("pause\n");
	//	return 0;
	//}

	////Run transpose kernel
	//transpose<<<1, 128>>>(d_atr,d_a,ARRAY_SIZE,ARRAY_SIZE);//addKernel


	////transfer the array from Device to Host(GPU->CPU) & check
	//cudaStatus = cudaMemcpy(h_check, d_atr, sizeof(float)*ARRAY_SIZE2, cudaMemcpyDeviceToHost);
	//cudaStatus = cudaGetLastError();
	//if(cudaStatus != cudaSuccess)
	//{
	//	printf("\nCuda Error:%s\n",cudaGetErrorString(cudaStatus));
	//	system("pause\n");
	//	return 0;
	//}

	////print out the transpose result
	//printf("\After transpose...\n");
	//for(long i = 0; i<ARRAY_SIZE2;i++){
	//	printf("%f\n", h_atr[i]);
	//}
	////print out the original Matrix A
	//printf("\nBefore transpose...\n");
	//for(long i = 0; i<ARRAY_SIZE2;i++){
	//	printf("%f\n", h_a[i]);
	//}
	printf("\n*****************End of transpose********************\n");





	//Run kernel function calculate the matrix-vector multiplication
	printf("\n\nRunning Kernel...\n\n");

	//Timer
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);
	cudaEventRecord(d_start,0); //mark event

	//Check
	//cudaError_t cudaState = cudaSuccess;
	cudaStatus = cudaSuccess;

	



    //MVKernel
	int nblocks= ARRAY_SIZE/512+1;
	CUParaSgemv<<<nblocks, 512>>>(d_a,d_b,d_c,ARRAY_SIZE);//addKernel
	//addKernel<<<1, ARRAY_SIZE>>>(d_a,d_b,d_c,ARRAY_SIZE);


	cudaThreadSynchronize();
	cudaEventRecord(d_stop,0);

	cudaEventSynchronize(d_stop);
	cudaEventElapsedTime(&cal_time,d_start,d_stop);

	printf( "\n%f milliseconds passed in GPU_CUParaSgemv\n", cal_time );

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(GPU calculation):%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}

	//printf( "\n%f milliseconds passed in calculation\n", time );

/*********************************************/
	
	printf("\n*********Copy Data to Host*********\n");
	
/**************GPU Memory copy out time********************/

	//Timer
	cudaEventCreate(&cuda_memcpyout_start);
	cudaEventCreate(&cuda_memcpyout_fin);
	cudaEventRecord(cuda_memcpyout_start,0); //mark event

	//transfer the array from Device to Host(GPU->CPU) & check
	cudaStatus = cudaMemcpy(h_c, d_c, sizeof(float)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaEventRecord(cuda_memcpyout_fin,0);

	cudaEventSynchronize(cuda_memcpyout_fin);
	cudaEventElapsedTime(&cudamemcpyout,cuda_memcpyout_start,cuda_memcpyout_fin);
	printf( "\n%f milliseconds passed in cuda memory copy out\n", cudamemcpyout );

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error:%s\n",cudaGetErrorString(cudaStatus));
		system("pause\n");
		return 0;
	}
	
/***********************************************/
	//system("pause");
/**************Print out the result********************/
	////print out the result array
	//for(long i = 0; i<ARRAY_SIZE;i++){
	//	printf("%f\n", h_c[i]);
	//}

/***********************************************/

	//free GPU memory allocation
	cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	//free Host memory allocation
	free(h_atr);
	free(h_a);
	free(h_b);
	free(h_c);

	system("pause");

    return 0;
}
