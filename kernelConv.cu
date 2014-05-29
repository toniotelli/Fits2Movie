#include "kernelConv.cuh"

__global__ void convert_fits_RGB(uint8_t *buff, double *data, int nx, int ny, double minD, double maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	double temp=0;
	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		temp = (data[CC]-minD)/(maxD-minD)*255;
		buff[3*CC]=(1.44068*temp > 255) ? 1.44068*temp : 255;
		buff[3*CC+1]=temp;
		buff[3*CC+2]=(temp <= 190) ? 0 : 3.92308*temp;
	}
}
void launchConvertion(uint8_t *buff, void *data, int nx, int ny, int minD, int maxD){
	dim3 dimB(BLOCKX,BLOCKY);
	dim3 dimG(nx/BLOCKX,ny/BLOCKY);

	printf("%i,%i\n",nx,ny);
	printf("%i,%i\n",nx/32,ny/32);

	double minDD=minD;
	double maxDD=maxD;
	printf("%i,%i\n",minD,minD);
	printf("%f,%f\n",minDD,maxDD);

	// lauch kernel
	convert_fits_RGB<<<dimG,dimB>>>(buff,(double *)data,nx,ny,minDD,maxDD);
	cudaDeviceSynchronize();
	check_CUDA_error("Convertion");
}

// Cuda handling
void check_CUDA_error(const char *message){
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
		printf("ERROR: %s: %s\n", message, cudaGetErrorString(error) );
		exit(-1);
	}
}
int checkCudaDevice(){
	int NBCudaDev=0;
	int devN=0;
	cudaGetDeviceCount(&NBCudaDev);

	cudaSetDevice(devN);
	cudaGetDevice(&devN);
	cudaDeviceReset();

	cudaSetDeviceFlags(cudaDeviceMapHost);
	printf("\nThere is %i CUDA Device using %i",NBCudaDev,devN);
	cudaDeviceProp dprop;
	cudaGetDeviceProperties(&dprop, devN);
	printf(" : %s\n", dprop.name);
	printf("Can Map host Mem : %i\n", dprop.canMapHostMemory);

	// Show device properties
	printf("Max Treads by block = %i.\n",dprop.maxThreadsPerBlock);
	printf("Max Grid Size X = %i.\n\n",dprop.maxGridSize[1]);
	return NBCudaDev;
}
