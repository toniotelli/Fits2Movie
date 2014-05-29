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

__global__ void convert_fits_RGB(uint8_t *buff, double *data, int nx, int ny, double minD, double maxD);
void launchConvertion(uint8_t *buff, void *data, int nx, int ny, double minD, double maxD);

// Cuda handling
void check_CUDA_error(const char *message);
int checkCudaDevice();

#endif
