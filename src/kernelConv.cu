#include "kernelConv.cuh"

// Inline functions in double
// Red Temperature (ind = r 255, g 120, b 190
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
		case 94:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)c3(temp);
			buff[3*CCOk+2]=(uint8_t)temp;
			break;
		case 131:
			buff[3*CCOk]=(uint8_t)g0(temp,120);
			buff[3*CCOk+1]=(uint8_t)r0(temp,255);
			buff[3*CCOk+2]=(uint8_t)r0(temp,255);
			break;
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 193:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 211:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 355:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)c1(temp);
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
		case 94:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)c3(temp);
			buff[3*CCOk+2]=(uint8_t)temp;
			break;
		case 131:
			buff[3*CCOk]=(uint8_t)g0(temp,120);
			buff[3*CCOk+1]=(uint8_t)r0(temp,255);
			buff[3*CCOk+2]=(uint8_t)r0(temp,255);
			break;
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 193:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 211:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 355:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)c1(temp);
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
		case 94:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)c3(temp);
			buff[3*CCOk+2]=(uint8_t)temp;
			break;
		case 131:
			buff[3*CCOk]=(uint8_t)g0(temp,120);
			buff[3*CCOk+1]=(uint8_t)r0(temp,255);
			buff[3*CCOk+2]=(uint8_t)r0(temp,255);
			break;
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 193:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 211:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 355:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)c1(temp);
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
		case 94:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)c3(temp);
			buff[3*CCOk+2]=(uint8_t)temp;
			break;
		case 131:
			buff[3*CCOk]=(uint8_t)g0(temp,120);
			buff[3*CCOk+1]=(uint8_t)r0(temp,255);
			buff[3*CCOk+2]=(uint8_t)r0(temp,255);
			break;
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 193:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 211:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 355:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)c1(temp);
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
		case 94:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)c3(temp);
			buff[3*CCOk+2]=(uint8_t)temp;
			break;
		case 131:
			buff[3*CCOk]=(uint8_t)g0(temp,120);
			buff[3*CCOk+1]=(uint8_t)r0(temp,255);
			buff[3*CCOk+2]=(uint8_t)r0(temp,255);
			break;
		case 171:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 193:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 211:
			buff[3*CCOk]=(uint8_t)c1(temp);
			buff[3*CCOk+1]=(uint8_t)temp;
			buff[3*CCOk+2]=(uint8_t)c2(temp);
			break;
		case 304:
			buff[3*CCOk]=(uint8_t)r0(temp,255);
			buff[3*CCOk+1]=(uint8_t)g0(temp,120);
			buff[3*CCOk+2]=(uint8_t)b0(temp,190);
			break;
		case 355:
			buff[3*CCOk]=(uint8_t)c2(temp);
			buff[3*CCOk+1]=(uint8_t)(temp);
			buff[3*CCOk+2]=(uint8_t)c1(temp);
			break;
		}
	}
}

// Alloc on cuda
void *allocData(size_t size_data){
	void *data;
	cudaMalloc((void **)&data,size_data);
	check_CUDA_error("Couldn't allocate data!");

	return data;
}
void freeData(void *data){
	cudaFree(data);
}

// launch Convertion
void launchConvertion(uint8_t *buff, void *data, int bitpix, int nx, int ny, double minD, double maxD, int wave){
	// Need to check if nx and ny are a multiple of BLOCK
	int NX= nx/BLOCKX + nx % BLOCKX;
	int NY= ny/BLOCKY + ny % BLOCKY;

	dim3 dimB(BLOCKX,BLOCKY);
	dim3 dimG(NX/BLOCKX,NY/BLOCKY);

	// printf("%i,%i\n",nx,ny);
	// printf("%i,%i\n",nx % BLOCKX,ny % BLOCKY);

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
void checkCudaDevice(){
	int NBCudaDev=0;
	int devN=0;
	cudaGetDeviceCount(&NBCudaDev);
	printf("Checking \033[32m%i\033[0m CUDA Devices: \n",NBCudaDev);
	if (NBCudaDev > 1){
		// should get the one with the most free space
		size_t *freMem,*totMem;
		size_t maxFree;
		freMem=(size_t *)malloc(NBCudaDev*sizeof(size_t));
		totMem=(size_t *)malloc(NBCudaDev*sizeof(size_t));
		for (int i=0; i<NBCudaDev; i++){
			cudaSetDevice(i);
			cudaMemGetInfo(&(freMem[i]),&(totMem[i]));
			printf("----Device %i has %zu MB free\n",i,freMem[i]/1024/1024);
			if (i == 0) maxFree=freMem[i];
			if (freMem[i] > maxFree) devN=i;
			cudaDeviceReset();
		}

		free(freMem);
		free(totMem);
	}
	
	// Set, check and reset CUDA Device
	size_t f,t;
	cudaSetDevice(devN);
	cudaGetDevice(&devN);
	cudaMemGetInfo(&f,&t);
	cudaDeviceReset();

	// Get Some information
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaDeviceProp dprop;
	cudaGetDeviceProperties(&dprop, devN);

	printf("CUDA Device %i",devN);
	printf(" : \033[34m%s\033[0m with \033[32m%zu\033[0m/\033[31m%zu\033[0m MB free.\n",dprop.name,f/1024/1024,t/1024/1024);
	printf("Can Map host Mem : %i\n", dprop.canMapHostMemory);

	// Show device properties
	printf("Max Treads by block = %i.\n",dprop.maxThreadsPerBlock);
	printf("Max Grid Size X = %i.\n",dprop.maxGridSize[1]);
}
