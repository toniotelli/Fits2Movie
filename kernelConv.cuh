//
//  kernelConv.cuh.h
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#ifndef Fits2Movie_kernelConv_cuh
#define Fits2Movie_kernelConv_cuh

#define BLOCKX 16
#define BLOCKY 16

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <math_functions.h>

__global__ void convert_fits_RGB(uint8_t *buff, void *data, int nx, int minD, int maxD);
void launchConvertion(uint8_t *buff, void *data, int nx, int ny, int minD, int maxD);

// Cuda handling
void check_CUDA_error(const char *message);
int checkCudaDevice();

#endif
