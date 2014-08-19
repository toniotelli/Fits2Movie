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

	struct arguments argm;
	// Init default variables
	argm.hFlag=false;
	argm.scale=false;
	argm.resize=false;
	argm.fpsU=false;

	// Time For benchMark
	time_t start,stop;
	clock_t ticks;

	// Start the event record
	time(&start);

	if (argc == 1) {
		printUsage(argv[0]);
		return 1;
	} else {
	error = parseCmdLine(argc,argv,optString,&argm);
		if (error != 0){
			printUsage(argv[0]);
			return 1;
		}
		
		if (!argm.fpsU){
			argm.fps=25;
		}
		if (argm.scale){
			dmin=argm.dMin;
			dmax=argm.dMax;
		} else {
			argm.dMin=0;
			argm.dMax=0;
		}
	}

	system("clear");
	printf("Terminal size = [%i,%i]",ws.ws_col,ws.ws_row);
	printHead("Welcome to Fits2Movie!",ws.ws_col);
	printf("Number of files = \033[32m%i\033[0m\n",argc);
	printf("Save movie in : \033[32m%s\033[0m\n",argm.output);
	printf("First fit files = \033[32m%s\033[0m\n",argv[argm.itStart]);
	printf("FPS = \033[32m%i\033[0m\n",argm.fps);
	if (argm.scale) printf("Scale = \033[32m%lf,%lf\033[0m\n",argm.dMin,argm.dMax);
	if (argm.resize) printf("Resize = \033[32m%i,%i\033[0m\n",argm.NX,argm.NY);

	// Cuda
	printHead("CUDA Capabilities",ws.ws_col);
	checkCudaDevice();

	// Fits Variables
	printHead("Fits Parameters",ws.ws_col);	

	// Declare the Needed size
	int status=0;
	int bitpix=0;
	int iS[2]={0,0};
	int tS[2]={0,0};
	int fS[2]={0,0};

	int min=0,max=0;
	int wave = 0;

	// Get image dimension
	status=getImageSize(argv[argm.itStart],&bitpix,iS,&min,&max,&wave);
	if (status != 0) {
		fits_report_error(stderr,status);
		exit(-1);
	}
	// print a short summary
	printf("BITPIX = %i\n",bitpix);
	printf("Image Size = [%i,%i]\n", iS[0],iS[1]);
	printf("Wavelenght = %i Angstrom\n",wave);

	// define size of temporary and final frame
	argm.padding=checkImgSize(iS,tS);
	printf("argm.padding = %i",argm.padding);

	// Check if images needs to be padded
	// if ((tS[0] != iS[0] || tS[1] != iS[1]) && (tS[0] !=0 || tS[1] != 0)) argm.padding=true;
	if (argm.padding) printf("\033[1;33mWarning\033[m : Images will be padded\n");
	if (!argm.resize) {
		if (argm.padding) {
			fS[0]=tS[0];
			fS[1]=tS[1];
		} else {
			fS[0]=iS[0];
			fS[1]=iS[1];
		}
	} else {
		fS[0]=argm.NX;
		fS[1]=argm.NY;
	}

	// Check value
	printf("Input Size: [%i,%i]\n",iS[0],iS[1]);
	printf("Tempo Size: [%i,%i]\n",tS[0],tS[1]);
	printf("Final Size: [%i,%i]\n",fS[0],fS[1]);

	// Put an exit choice
	char ch[]="";
	printf("Continue: [Y/n]:");
	scanf("%c",ch);
	if (ch[0] == 'n') return EXIT_FAILURE;

	// AVCodec variable
	printHead("FFMpeg",ws.ws_col);
	struct AVFormatContext *oc;
	struct AVCodec *avCodec;
	struct AVStream *avStream;
	struct AVFrame *fRGB,*fTemp;
	struct AVFrame *fFinal;

	// Declare the frameBuffer for encoding
	size_t sRGB,sTemp,sFinal;
	uint8_t *hbRGB,*hbTemp,*hbFinal;

	// Compute the size of needed buffer for libav
	sRGB=computeRGBSize(tS[0],tS[1]);
	sTemp=computeYUVSize(tS[0],tS[1]);
	sFinal=computeYUVSize(fS[0],fS[1]);

	// Alloc the necessary memory space
	hbRGB=(uint8_t *)malloc(sRGB);
	if (argm.resize) hbTemp=(uint8_t *)malloc(sTemp);
	hbFinal=(uint8_t *)malloc(sFinal);
	
	// Init avcodec
	av_register_all();
	av_log_set_level(AV_LOG_MAX_OFFSET);
	// av_log_set_level(AV_LOG_DEBUG);
	// av_log_set_level(AV_LOG_INFO);
	// av_log_set_level(AV_LOG_ERROR);


	// Open Movie file and alloc necessary stuff
	remove(argm.output);
	openFormat(argm.output, &oc);
	openStream(oc, &avCodec, &avStream, fS[0], fS[1], argm.fps);
	openCodec(&avCodec, avStream);
	if (argm.resize) {
		allocFrames(avStream, &fRGB, &fTemp, hbRGB, hbTemp, tS[0], tS[1]);
		allocFrameConversion(&fFinal,hbFinal,fS[0],fS[1]);
	} else {
		allocFrames(avStream, &fRGB, &fFinal, hbRGB, hbFinal, fS[0], fS[1]);
	}
	writeHeader(argm.output, oc);

	// Alloc buffer fits
	void *data=NULL,*dataTemp=NULL;
	size_t sData=0,rData=0;
	sData=allocDataType(&data,bitpix,iS[0],iS[1]);
	if (argm.padding) rData=allocDataType(&dataTemp,bitpix,tS[0],tS[1]);
	printf("buffer size= %zu, data size = %zu, padded data size = %zu\n",sRGB,sData,rData);

	// Alloc buffer and data
	uint8_t *bufRGB=(uint8_t *)allocData(sRGB);
	void *dData=NULL,*dDataTemp=NULL;
	if (argm.padding){
		dDataTemp=allocData(sData);
		dData=allocData(rData);
	} else {
		dData=allocData(sData);
	}

	// Run loop
	printHead("Converting",ws.ws_col);
	// printf("\n\033[34m\\********************** Converting **********************/\033[0m\n");
	for (int i=argm.itStart; i<argc; i++) {
		// printf("\033[7Fits: %s\n",argv[i]);
		printf("Fits: %s\n",argv[i]);
		status=readFits(argv[i],data,bitpix,iS,&min,&max);

		// copy data to device
		if (argm.padding){
			cudaMemcpy(dDataTemp, data, sData, cudaMemcpyHostToDevice);
			check_CUDA_error("Copying H to D");

			// launch the padding kernel
			launchPadding(dData,dDataTemp,bitpix,tS[0],tS[1],iS[0],iS[1]);
		} else {
			cudaMemcpy(dData, data, sData, cudaMemcpyHostToDevice);
			check_CUDA_error("Copying H to D");
		}

		if (!argm.scale){
			dmin=min;
			dmax=max;
		} else {
			dmin=argm.dMin;
			dmax=argm.dMax;
		}
		printf("User[min,max]=[%lf,%lf], [%i,%i]\n",dmin,dmax,tS[0],tS[1]);

		// launch the process
		launchConvertion(bufRGB, dData, bitpix, tS[0], tS[1], dmin, dmax, wave);

		// copy back buffRGB to host
		cudaMemcpy(hbRGB,bufRGB,sRGB,cudaMemcpyDeviceToHost);
		check_CUDA_error("Copying D to H");

		// Rescale and encode frame
		if (argm.resize){
			rescaleRGBToYUV(fRGB,fTemp,hbRGB,hbTemp,sRGB);
			rescaleYUV(fTemp,fFinal,hbTemp,hbFinal,sTemp);
		} else {
			rescaleRGBToYUV(fRGB,fFinal,hbRGB,hbFinal,sRGB);
		}
		encodeOneFrameYUV(oc,avStream,fFinal,i);

		// Progress and cpu ticks
		printProgress(i, argc-1,ws.ws_col);
		// printf("\033[8");
		// if (i < argc-1) printf("\033[7A");
		// if (i < argc-1) {
		// 	if (argm.padding) {
		// 		printf("\033[5A");
		// 	} else {
		// 		printf("\033[4A");
		// 	}
		// }
		ticks=clock();
	}
	
	// Compute time elapse on initialization
	time(&stop);
	printf("Used %0.2f seconds of CPU time. \n", (double)ticks/CLOCKS_PER_SEC);
	printf("Time spent for %i fits of [%i,%i]: \033[31m%f [min]\033[0m\n",argc,iS[0],iS[1],difftime(stop,start)/60.0);
	
	// Free Graphic Memory
	if (argm.padding) freeData(dDataTemp);
	freeData(bufRGB);
	freeData(dData);
	free(data);

	// dealloc movie files
	av_write_trailer(oc);
	avcodec_close(avStream->codec);
	av_free(avStream);
	avio_close(oc->pb);
	deallocFrames(fRGB, fFinal, hbRGB, hbFinal);
	if (argm.resize) deallocFrameConversion(fTemp,hbTemp);

    cudaDeviceReset();

    return 0;
}
