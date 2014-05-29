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
//#include <libavutil/opt.h>
//#include <libavcodec/avcodec.h>
//#include <libavformat/avformat.h>
//#include <libavutil/avconfig.h>
//#include <libavutil/channel_layout.h>
//#include <libavutil/common.h>
//#include <libavutil/imgutils.h>
//#include <libavutil/mathematics.h>
//#include <libavutil/samplefmt.h>
//#include <libswscale/swscale.h>
//#include <libswresample/swresample.h>
//#include <libavutil/timestamp.h>
#include "aviFunction.h"
#include "fitsFunction.h"

}

int main(int argc, const char * argv[]){
    printf("Welcome to %s!\n",argv[0]);
    printf("Number of files = %i\n",argc);
    if (argc == 1) {
        printf("Usage : %s out.mkv *.fits\n",argv[0]);
        return -1;
    }
    // Cuda
    int nbCuda=0;
    nbCuda=checkCudaDevice();
    printf("There is %i devices\n",nbCuda);

    // Fits Variables
    int status=0;
    int imgSize[]={0,0,0};
    int min=0,max=0;
    
    // Get image dimension
    status=getImageSize(argv[2],imgSize,&min,&max);

    // AVCodec variable
    struct AVFormatContext *oc;
    struct AVCodec *avCodec;
    struct AVStream *avStream;
    struct AVFrame *frameRGB,*frameYUV;

    // Alloc the frameBuffer for encoding
    size_t bRGB=3*imgSize[1]*imgSize[2]*sizeof(uint8_t);
    size_t bYUV=2*imgSize[1]*imgSize[2]*sizeof(uint8_t);
    uint8_t *hbRGB,*hbYUV;
    hbRGB=(uint8_t *)malloc(bRGB);
    hbYUV=(uint8_t *)malloc(bYUV);

    // Init avcodec
    av_register_all();
    av_log_set_level(AV_LOG_INFO);
    
    // Open Movie file and alloc necessary stuff
    remove(argv[1]);
    openFormat(argv[1], &oc);
    openStream(oc, &avCodec, &avStream, imgSize[1], imgSize[2], 30);
    openCodec(&avCodec, avStream);
    allocFrames(avStream, &frameRGB, &frameYUV, hbRGB, hbYUV);
    writeHeader(argv[1], oc);
    printf("Using %s: %s\nCodec: %s\n",oc->oformat->name,oc->oformat->long_name,avcodec_get_name(oc->oformat->video_codec));
    
    // Get a pos just to check
    int x=2000,y=1500;
    int pos=y*imgSize[1]+x;
    
    // Alloc buffer fits
    void *data=NULL;
    size_t sData=0;
    sData=allocDataType(&data,imgSize[0],imgSize[1],imgSize[2]);

    // Test if cuda works
    printf("buffer size= %zu, data size = %zu",bRGB,sData);
    uint8_t *dbRGB;
    cudaMalloc((void **)&dbRGB,bRGB);
    void *dData;
    cudaMalloc((void **)&dData, sData);
    
    for (int i=2; i<argc; i++) {
        printf("Fits: %s\n",argv[i]);
        status=readFits(argv[i],data, imgSize,&min,&max);
        printf("data[%i,%i]=%f\n",y,x,((double *)data)[pos]);
        printf("data[min,max]=[%f,%f]\n",(double)min,(double)max);
        
        // copy data to device
        cudaMemcpy(dData, data, sData, cudaMemcpyHostToDevice);
        check_CUDA_error("Copying H to D");
        
        // launch the process
        launchConvertion(dbRGB, dData, imgSize[1], imgSize[2], min, max);
        
        // copy back buffRGB to host
        cudaMemcpy(hbRGB,dbRGB,bRGB,cudaMemcpyDeviceToHost);
        check_CUDA_error("Copying D to H");
        
        // Rescale and encode frame
        rescaleRGBToYUV(frameRGB,frameYUV,hbRGB,hbYUV,bRGB);
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

    return 0;
}

