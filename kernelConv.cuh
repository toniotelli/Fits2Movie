//
//  kernelConv.cuh.h
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#ifndef Fits2Movie_kernelConv_cuh
#define Fits2Movie_kernelConv_cuh

#ifndef __APPLE__
	#define BLOCKX 32
	#define BLOCKY 32
#else
	#define BLOCKX 16
	#define BLOCKY 16
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fitsio.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <math_functions.h>

// Inline functions in double
// Red Temperature
__device__ inline double r0(double temp, int ind);
__device__ inline double g0(double temp, int ind);
__device__ inline double b0(double temp, int ind);


// Convertion to 171 colormap
__global__ void convert_fits_RGB_double_171(uint8_t *buff, double *data, int nx, int ny, double minD, double maxD);
__global__ void convert_fits_RGB_float_171(uint8_t *buff, float *data, int nx, int ny, float minD, float maxD);
__global__ void convert_fits_RGB_long_171(uint8_t *buff, long *data, int nx, int ny, long minD, long maxD);
__global__ void convert_fits_RGB_shortInt_171(uint8_t *buff, short int *data, int nx, int ny, short int minD, short int maxD);
__global__ void convert_fits_RGB_uchar_171(uint8_t *buff, unsigned char *data, int nx, int ny, unsigned char minD, unsigned char maxD);

// launch Convertion
void launchConvertion(uint8_t *buff, void *data, int bitpix, int nx, int ny, double minD, double maxD);

// Cuda handling
void check_CUDA_error(const char *message);
int checkCudaDevice();

#endif
