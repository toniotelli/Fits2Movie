#include "kernelConv.cuh"

// Inline functions in double
// Red Temperature
__device__ inline double r0(double temp, int ind){ return ((1.44068*temp > ind) ? 255 : 1.44068*temp);}
__device__ inline double g0(double temp, int ind){ return ((temp <= ind) ? 0 : 1.88889*temp);}
__device__ inline double b0(double temp, int ind){ return ((temp <= ind) ? 0 : 3.92308*temp);}
// c0 = temp
__device__ inline double c1(double temp){ return sqrtf(temp)*sqrtf(255);};
__device__ inline double c2(double temp){ return (temp*temp)/255;};
__device__ inline double c3(double temp){ return ((sqrtf(temp)*sqrtf(255)+(temp*temp)/255/2))/255;};

// Convertion to 171 colormap
__global__ void convert_fits_RGB_double(uint8_t *buff, double *data, int wave, int nx, int ny, double minD, double maxD){
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
		switch(wave){
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		}
	}
}
__global__ void convert_fits_RGB_float(uint8_t *buff, float *data, int wave, int nx, int ny, float minD, float maxD){
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
		switch(wave){
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		}
	}
}
__global__ void convert_fits_RGB_long(uint8_t *buff, long *data, int wave, int nx, int ny, long minD, long maxD){
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
		switch(wave){
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		}
	}
}
__global__ void convert_fits_RGB_shortInt(uint8_t *buff, short int *data, int wave, int nx, int ny, short int minD, short int maxD){
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
		switch(wave){
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		}
	}
}
__global__ void convert_fits_RGB_uchar(uint8_t *buff, unsigned char *data, int wave, int nx, int ny, unsigned char minD, unsigned char maxD){
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
		switch(wave){
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		}
	}
}

// launch Convertion
void launchConvertion(uint8_t *buff, void *data, int bitpix, int nx, int ny, double minD, double maxD, int wave){
	dim3 dimB(BLOCKX,BLOCKY);
	dim3 dimG(nx/BLOCKX,ny/BLOCKY);

	printf("%i,%i\n",nx,ny);
	printf("%i,%i\n",nx/32,ny/32);

	// lauch kernel
	switch(bitpix){
	case BYTE_IMG:
		convert_fits_RGB_uchar<<<dimG,dimB>>>(buff,(unsigned char *)data,wave,nx,ny,(unsigned char)minD,(unsigned char)maxD);
		break;
	case SHORT_IMG:
		convert_fits_RGB_shortInt<<<dimG,dimB>>>(buff,(short int *)data,wave,nx,ny,(short int)minD,(short int)maxD);
		break;
	case LONG_IMG:
		convert_fits_RGB_long<<<dimG,dimB>>>(buff,(long *)data,wave,nx,ny,(long)minD,(long)maxD);
		break;
	case FLOAT_IMG:
		convert_fits_RGB_float<<<dimG,dimB>>>(buff,(float *)data,wave,nx,ny,(float)minD,(float)maxD);
		break;
	case DOUBLE_IMG:
		convert_fits_RGB_double<<<dimG,dimB>>>(buff,(double *)data,wave,nx,ny,(double)minD,(double)maxD);
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
