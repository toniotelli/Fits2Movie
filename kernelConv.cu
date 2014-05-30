#include "kernelConv.cuh"

// Inline functions in double
// Red Temperature
__device__ inline double r0(double temp, int ind){ return ((1.44068*temp > ind) ? 255 : 1.44068*temp);}
__device__ inline double g0(double temp, int ind){ return ((temp <= ind) ? 0 : 1.88889*temp);}
__device__ inline double b0(double temp, int ind){ return ((temp <= ind) ? 0 : 3.92308*temp);}

// Convertion to 171 colormap
__global__ void convert_fits_RGB_double_171(uint8_t *buff, double *data, int nx, int ny, double minD, double maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	int CCOk=(ny-1-y)*nx+x;
	double temp=0;
//	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		if (data[CC] < minD) {
			temp =0;
		} else if (data[CC] > maxD){
			temp =255;
		} else {
			temp = (data[CC]-minD)/(maxD-minD)*255;
		}
		// attempt to emule loadct data
		buff[3*CCOk]=(uint8_t)r0(temp,255);
		buff[3*CCOk+1]=(uint8_t)(temp);
		buff[3*CCOk+2]=(uint8_t)b0(temp,255);
	}
}
__global__ void convert_fits_RGB_float_171(uint8_t *buff, float *data, int nx, int ny, float minD, float maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	int CCOk=(ny-1-y)*nx+x;
	double temp=0;
//	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		if (data[CC] < minD) {
			temp =0;
		} else if (data[CC] > maxD){
			temp =255;
		} else {
			temp = (data[CC]-minD)/(double)(maxD-minD)*255;
		}
		// attempt to emule loadct data
		buff[3*CCOk]=(uint8_t)r0(temp,255);
		buff[3*CCOk+1]=(uint8_t)(temp);
		buff[3*CCOk+2]=(uint8_t)b0(temp,255);
	}
}
__global__ void convert_fits_RGB_long_171(uint8_t *buff, long *data, int nx, int ny, long minD, long maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	double temp=0;
//	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		if (data[CC] < minD) {
			temp =0;
		} else if (data[CC] > maxD){
			temp =255;
		} else {
			temp = (data[CC]-minD)/(double)(maxD-minD)*255;
		}
		// attempt to emule loadct data
		buff[3*CC]=(uint8_t)r0(temp,255);
		buff[3*CC+1]=(uint8_t)(temp);
		buff[3*CC+2]=(uint8_t)b0(temp,255);
	}
}
__global__ void convert_fits_RGB_shortInt_171(uint8_t *buff, short int *data, int nx, int ny, short int minD, short int maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	int CCOk=(ny-1-y)*nx+x;
	double temp=0;
//	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		if (data[CC] < minD) {
			temp =0;
		} else if (data[CC] > maxD){
			temp =255;
		} else {
			temp = (data[CC]-minD)/(double)(maxD-minD)*255;
		}
		// attempt to emule loadct data
		buff[3*CCOk]=(uint8_t)r0(temp,255);
		buff[3*CCOk+1]=(uint8_t)(temp);
		buff[3*CCOk+2]=(uint8_t)b0(temp,255);
	}
}
__global__ void convert_fits_RGB_uchar_171(uint8_t *buff, unsigned char *data, int nx, int ny, unsigned char minD, unsigned char maxD){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int CC=y*nx+x;
	int CCOk=(ny-1-y)*nx+x;
	double temp=0;
//	printf("data[%i]=%f\n",CC,data[CC]);

	if (CC < nx*ny){
		if (data[CC] < minD) {
			temp =0;
		} else if (data[CC] > maxD){
			temp =255;
		} else {
			temp = (data[CC]-minD)/(double)(maxD-minD)*255;
		}
		// attempt to emule loadct data
		buff[3*CCOk]=(uint8_t)r0(temp,255);
		buff[3*CCOk+1]=(uint8_t)(temp);
		buff[3*CCOk+2]=(uint8_t)b0(temp,255);
	}
}

// launch Convertion
void launchConvertion(uint8_t *buff, void *data, int bitpix, int nx, int ny, double minD, double maxD){
	dim3 dimB(BLOCKX,BLOCKY);
	dim3 dimG(nx/BLOCKX,ny/BLOCKY);

	printf("%i,%i\n",nx,ny);
	printf("%i,%i\n",nx/32,ny/32);

	// lauch kernel
	switch(bitpix){
	case BYTE_IMG:
		convert_fits_RGB_uchar_171<<<dimG,dimB>>>(buff,(unsigned char *)data,nx,ny,(unsigned char)minD,(unsigned char)maxD);
		break;
	case SHORT_IMG:
		convert_fits_RGB_shortInt_171<<<dimG,dimB>>>(buff,(short int *)data,nx,ny,(short int)minD,(short int)maxD);
		break;
	case LONG_IMG:
		convert_fits_RGB_long_171<<<dimG,dimB>>>(buff,(long *)data,nx,ny,(long)minD,(long)maxD);
		break;
	case FLOAT_IMG:
		convert_fits_RGB_float_171<<<dimG,dimB>>>(buff,(float *)data,nx,ny,(float)minD,(float)maxD);
		break;
	case DOUBLE_IMG:
		convert_fits_RGB_double_171<<<dimG,dimB>>>(buff,(double *)data,nx,ny,(double)minD,(double)maxD);
		break;
	}
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
	if (NBCudaDev > 1){
		devN=1;
	}

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
