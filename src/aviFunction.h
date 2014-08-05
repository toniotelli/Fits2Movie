//
//  aviFunction.h
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#ifndef Fits2Movie_aviFunction_h
#define Fits2Movie_aviFunction_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avconfig.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/timestamp.h>

// allocation frames
void allocFrameConversion(AVFrame **frameYUV,uint8_t *buffYUV,int width,int height);
void allocFrames(AVStream *st, AVFrame **frameRGB, AVFrame **frameYUV, uint8_t *buffRGB, uint8_t *buffYUV,int width, int height);
void deallocFrames(AVFrame *frameRGB, AVFrame *frameYUV, uint8_t *buffRGB, uint8_t *buffYUV);
void deallocFrameConversion(AVFrame *frameYUV,uint8_t *buffYUV);

// Movie header and allocation
int write_frame(AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt);
void writeHeader(const char *filename,AVFormatContext *oc);
void openFormat(const char * filename,AVFormatContext **oc);
void openStream(AVFormatContext *oc, AVCodec **codec, AVStream **st, int width, int height, int fps);
void openCodec(AVCodec **codec, AVStream *st);

// Image processing
void rescaleRGBToYUV(AVFrame *framergb, AVFrame *frameyuv, uint8_t *buffrgb, uint8_t *buffyuv, size_t sizeB1);
void rescaleYUV(AVFrame *frameyuv0, AVFrame *frameyuv1, uint8_t *buffrgb0, uint8_t *buffyuv1, size_t sizeB1);
void encodeOneFrameYUV(AVFormatContext *oc,AVStream *st,AVFrame *frmyuv, int i);

#endif
