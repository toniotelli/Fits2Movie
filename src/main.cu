//
//  main.c
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fitsio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "kernelConv.cuh"

extern "C" {
#include "parserCmdLine.h"
#include "fitsFunction.h"
#include "aviFunction.h"
}

int main(int argc, char * argv[]){
	// get terminal size
	struct winsize ws;
   	ioctl(0, TIOCGWINSZ, &ws);

	double dmin=0.0,dmax=0.0;
	int error=0;

	struct arguments arguments;
	// Init default variables
	arguments.hFlag=false;
	arguments.scale=false;
	arguments.resize=false;
	arguments.fpsU=false;

	// Time For benchMark
	time_t start,stop;
	clock_t ticks;

	// Start the event record
	time(&start);

	if (argc == 1) {
		printUsage(argv[0]);
		return 1;
	} else {
	error = parseCmdLine(argc,argv,optString,&arguments);
		if (error != 0){
			printUsage(argv[0]);
			return 1;
		}
		
		if (!arguments.fpsU){
			arguments.fps=25;
		}
		if (arguments.scale){
			dmin=arguments.dMin;
			dmax=arguments.dMax;
		} else {
			arguments.dMin=0;
			arguments.dMax=0;
		}
	}

	system("clear");
	printf("Terminal size = [%i,%i]",ws.ws_col,ws.ws_row);
	printHead("Welcome to Fits2Movie!",ws.ws_col);
	printf("Number of files = \033[32m%i\033[0m\n",argc);
	printf("Save movie in : \033[32m%s\033[0m\n",arguments.output);
	printf("First fit files = \033[32m%s\033[0m\n",argv[arguments.itStart]);
	printf("FPS = \033[32m%i\033[0m\n",arguments.fps);
	if (arguments.scale) printf("Scale = \033[32m%lf,%lf\033[0m\n",arguments.dMin,arguments.dMax);
	if (arguments.resize) printf("Resize = \033[32m%i,%i\033[0m\n",arguments.NX,arguments.NY);

	// Cuda
	printHead("CUDA Capabilities",ws.ws_col);
	checkCudaDevice();

	// Fits Variables
	printHead("Fits Parameters",ws.ws_col);	
	int status=0;
	int imgSize[3]={0,0,0};
	int *newSize;
	int min=0,max=0;
	int wave = 0;

	// Get image dimension
	status=getImageSize(argv[arguments.itStart],imgSize,&min,&max,&wave);
	if (status != 0) {
		fits_report_error(stderr,status);
		exit(-1);
	}
	printf("BITPIX = %i\n",imgSize[0]);
	printf("Image Size = [%i,%i]\n", imgSize[1],imgSize[2]);
	printf("Wavelenght = %i Angstrom\n",wave);

	// check if image dimension is a power of 2
	newSize=checkImgSize(imgSize);
	if (newSize[1] != imgSize[1] || newSize[2] != imgSize[2]) arguments.padding=true;
	// if (newSize[1] != imgSize[1]) imgSize[1]=newSize[1];
	// if (newSize[2] != imgSize[2]) imgSize[2]=newSize[2];
	if (arguments.padding) printf("\033[1;33mWarning\033[m : Images will be padded\n");

	// AVCodec variable
	printHead("FFMpeg",ws.ws_col);
	struct AVFormatContext *oc;
	struct AVCodec *avCodec;
	struct AVStream *avStream;
	struct AVFrame *frameRGB,*frameYUV;
	struct AVFrame *frameYUVConv;

	// Alloc the frameBuffer for encoding
	size_t bRGB=rgbBuffSize(imgSize[1],imgSize[2]);
	size_t bYUV=yuvBuffSize(imgSize[1],imgSize[2]);
	size_t bYUVConv=0;
	if (arguments.resize) {
		bYUVConv=bYUV;
		bYUV=yuvBuffSize(arguments.NX,arguments.NY);
	}
	uint8_t *hbRGB,*hbYUV,*hbYUVConv;
	hbRGB=(uint8_t *)malloc(bRGB);
	hbYUV=(uint8_t *)malloc(bYUV);
	if (arguments.resize){
		hbYUVConv = (uint8_t *)malloc(bYUVConv);
	}

	// Init avcodec
	av_register_all();
	// av_log_set_level(AV_LOG_INFO);
	av_log_set_level(AV_LOG_ERROR);

	// Open Movie file and alloc necessary stuff
	remove(arguments.output);
	openFormat(arguments.output, &oc);
	if (arguments.resize){
		openStream(oc, &avCodec, &avStream, arguments.NX, arguments.NY, arguments.fps);
		openCodec(&avCodec, avStream);
		allocFrames(avStream, &frameRGB, &frameYUV, hbRGB, hbYUV, imgSize[1], imgSize[2]);
		allocFrameConversion(&frameYUVConv,hbYUVConv,arguments.NX,arguments.NY);
	} else {
		openStream(oc, &avCodec, &avStream, imgSize[1], imgSize[2], arguments.fps);
		openCodec(&avCodec, avStream);
		allocFrames(avStream, &frameRGB, &frameYUV, hbRGB, hbYUV, imgSize[1], imgSize[2]);
	}
	writeHeader(arguments.output, oc);

	// Alloc buffer fits
	void *data=NULL;
	size_t sData=0;
	sData=allocDataType(&data,imgSize[0],imgSize[1],imgSize[2]);

	// Test if cuda works
	printf("buffer size= %zu, data size = %zu\n",bRGB,sData);
	uint8_t *dbRGB;
	void *dData;

	// Alloc buffer and data
	dbRGB=(uint8_t *)allocData(bRGB);
	dData=allocData(sData);
	
	// Run loop
	printHead("Converting",ws.ws_col);
	// printf("\n\033[34m\\********************** Converting **********************/\033[0m\n");
	for (int i=arguments.itStart; i<argc; i++) {
		printf("Fits: %s\n",argv[i]);
		status=readFits(argv[i],data, imgSize,&min,&max);

		// copy data to device
		cudaMemcpy(dData, data, sData, cudaMemcpyHostToDevice);
		check_CUDA_error("Copying H to D");

		if (!arguments.scale){
			dmin=min;
			dmax=max;
		} else {
			dmin=arguments.dMin;
			dmax=arguments.dMax;
		}
		printf("User[min,max]=[%lf,%lf]\n",dmin,dmax);

		// launch the process
		launchConvertion(dbRGB, dData, imgSize[0], imgSize[1], imgSize[2], dmin, dmax, wave);

		// copy back buffRGB to host
		cudaMemcpy(hbRGB,dbRGB,bRGB,cudaMemcpyDeviceToHost);
		check_CUDA_error("Copying D to H");

		// Rescale and encode frame
		if (arguments.resize){
			rescaleRGBToYUV(frameRGB,frameYUVConv,hbRGB,hbYUVConv,bRGB);
			rescaleYUV(frameYUVConv,frameYUV,hbYUVConv,hbYUV,bYUV);
		} else {
			rescaleRGBToYUV(frameRGB,frameYUV,hbRGB,hbYUV,bRGB);
		}
		encodeOneFrameYUV(oc,avStream,frameYUV,i);
		
		// Progress and cpu ticks
		printProgress(i, argc-1,ws.ws_col);
		if (i < argc-1) printf("\033[4A");
		ticks=clock();
	}
	
	// Compute time elapse on initialization
	time(&stop);
	printf("Used %0.2f seconds of CPU time. \n", (double)ticks/CLOCKS_PER_SEC);
	printf("Time spent for %i fits of [%i,%i]: \033[31m%f [min]\033[0m\n",argc,imgSize[1],imgSize[2],difftime(stop,start)/60.0);
	
	// Free Graphic Memory
	freeData(dbRGB);
	freeData(dData);
	free(data);
	
	// dealloc movie files
	av_write_trailer(oc);
	avcodec_close(avStream->codec);
	av_free(avStream);
	avio_close(oc->pb);
	deallocFrames(frameRGB, frameYUV, hbRGB, hbYUV);
	if (arguments.resize) deallocFrameConversion(frameYUVConv,hbYUVConv);

    cudaDeviceReset();

    return 0;
}
