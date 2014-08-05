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
#include <unistd.h>
#include <fitsio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "kernelConv.cuh"

extern "C" {
#include "aviFunction.h"
#include "fitsFunction.h"
#include "parserCmdLine.h"
}

int main(int argc, char * argv[]){
	double dmin=0.0,dmax=0.0;
// printf("Welcome to %s!\n",argv[0]);

	arguments arguments;
	// Init default variables
	arguments.hFlag=false;
	arguments.scale=false;
	arguments.resize=false;
	arguments.fpsU=false;
	arguments.dMinMax[0]=dmin;
	arguments.dMinMax[1]=dmax;
	arguments.fps=25;

	
	if (argc == 1) {
		printUsage(argv[0]);
		return 1;
	} else {
	int error = parseCmdLine(argc,argv,optString,&arguments);
		if (error != 0){
			return 1;
		}

		// if (!arguments.fpsU){
		// 	arguments.fps=25;
		// }
		// if (arguments.scale){
		// 	dmin=arguments.dMinMax[0];
		// 	dmax=arguments.dMinMax[1];
		// } else {
		// 	arguments.dMinMax[0]=0.0;
		// 	arguments.dMinMax[1]=0.0;
		// }
	}

	printf("Number of files = %i\n",argc);
	printf("Save movie in : %s\n",arguments.output);
	printf("First fit files = %s\n",argv[arguments.itStart]);
	printf("FPS = %i\n",arguments.fps);

	// Cuda
	int nbCuda=0;
	nbCuda=checkCudaDevice();
	printf("There is %i devices\n",nbCuda);

	// Fits Variables
	int status=0;
	int imgSize[]={0,0,0};
	int min=0,max=0;
	int wave = 0;

	// Get image dimension
	status=getImageSize(argv[arguments.itStart],imgSize,&min,&max,&wave);
	if (status != 0) {
		fits_report_error(stderr,status);
		exit(-1);
	}

	// AVCodec variable
	printf("AV struct \n");
	struct AVFormatContext *oc;
	struct AVCodec *avCodec;
	struct AVStream *avStream;
	struct AVFrame *frameRGB,*frameYUV;
	struct AVFrame *frameYUVConv;


	// Alloc the frameBuffer for encoding
	size_t bRGB=3*imgSize[1]*imgSize[2]*sizeof(uint8_t);
	size_t bYUV=2*imgSize[1]*imgSize[2]*sizeof(uint8_t);
	size_t bYUVConv=0;
	if (arguments.resize == 1) {
		bYUVConv=bYUV;
		bYUV=2*arguments.NXNY[0]*arguments.NXNY[1]*sizeof(uint8_t);
	}
	uint8_t *hbRGB,*hbYUV,*hbYUVConv;
	hbRGB=(uint8_t *)malloc(bRGB);
	hbYUV=(uint8_t *)malloc(bYUV);
	if (arguments.resize == 1){
		hbYUVConv = (uint8_t *)malloc(bYUVConv);
	}

	// Init avcodec
	av_register_all();
	av_log_set_level(AV_LOG_INFO);

	// Open Movie file and alloc necessary stuff
	remove(arguments.output);
	openFormat(arguments.output, &oc);
	if (arguments.scale == 1){
		openStream(oc, &avCodec, &avStream, arguments.NXNY[0], arguments.NXNY[1], arguments.fps);
		openCodec(&avCodec, avStream);
		allocFrames(avStream, &frameRGB, &frameYUV, hbRGB, hbYUV, imgSize[1], imgSize[2]);
		allocFrameConversion(&frameYUVConv,hbYUVConv,arguments.NXNY[0],arguments.NXNY[1]);
	} else {
		openStream(oc, &avCodec, &avStream, imgSize[1], imgSize[2], arguments.fps);
		openCodec(&avCodec, avStream);
		allocFrames(avStream, &frameRGB, &frameYUV, hbRGB, hbYUV, imgSize[1], imgSize[2]);
	}
	writeHeader(arguments.output, oc);
	printf("Using %s: %s\nCodec: %s\n",oc->oformat->name,oc->oformat->long_name,avcodec_get_name(oc->oformat->video_codec));

	// Alloc buffer fits
	void *data=NULL;
	size_t sData=0;
	sData=allocDataType(&data,imgSize[0],imgSize[1],imgSize[2]);

	// Test if cuda works
	printf("buffer size= %zu, data size = %zu\n",bRGB,sData);
	uint8_t *dbRGB;
	cudaMalloc((void **)&dbRGB,bRGB);
	void *dData;
	cudaMalloc((void **)&dData, sData);

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
			dmin=arguments.dMinMax[0];
			dmin=arguments.dMinMax[1];
		}
		printf("data[min,max]=[%lf,%lf]\n",dmin,dmax);


		// launch the process
		launchConvertion(dbRGB, dData, imgSize[0], imgSize[1], imgSize[2], dmin, dmax, wave);

		// copy back buffRGB to host
		cudaMemcpy(hbRGB,dbRGB,bRGB,cudaMemcpyDeviceToHost);
		check_CUDA_error("Copying D to H");

		// Rescale and encode frame
		if (arguments.scale == 1){
			rescaleRGBToYUV(frameRGB,frameYUVConv,hbRGB,hbYUVConv,bRGB);
			rescaleYUV(frameYUVConv,frameYUV,hbYUVConv,hbYUV,bYUV);
		} else {
			rescaleRGBToYUV(frameRGB,frameYUV,hbRGB,hbYUV,bRGB);
		}
		encodeOneFrameYUV(oc,avStream,frameYUV,i);
	}
	cudaFree(dbRGB);
	cudaFree(dData);
	free(data);
	
	// dealloc movie files
	av_write_trailer(oc);
	avcodec_close(avStream->codec);
	av_free(avStream);
	avio_close(oc->pb);
	deallocFrames(frameRGB, frameYUV, hbRGB, hbYUV);
	if (arguments.scale == 1) deallocFrameConversion(frameYUVConv,hbYUVConv);

	cudaDeviceReset();

	return 0;
}
