//
//  aviFunction.c
//  Fits2Movie
//
//  Created by Antoine Genetelli on 28/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#include "aviFunction.h"

// allocation frames
void allocFrameConversion(AVFrame **frameYUV,uint8_t *buffYUV,int width,int height){
	// YUV Frame
	*frameYUV=av_frame_alloc();
	(*frameYUV)->width=width;
	(*frameYUV)->height=height;
	(*frameYUV)->format=AV_PIX_FMT_YUV420P;

	// Alloc the images now
	int sB1=0;
	sB1=avpicture_fill((AVPicture *)*frameYUV, buffYUV,AV_PIX_FMT_YUV420P,(*frameYUV)->width, (*frameYUV)->height);
}
void allocFrames(AVStream *st, AVFrame **frameRGB, AVFrame **frameYUV, uint8_t *buffRGB, uint8_t *buffYUV,int width, int height){
	int sB1=0,sB2=0;

	// RGB Frame
	*frameRGB=av_frame_alloc();
	(*frameRGB)->width=width;
	(*frameRGB)->height=height;
	(*frameRGB)->format=AV_PIX_FMT_RGB24;

	// YUV Frame
	*frameYUV=av_frame_alloc();
	(*frameYUV)->width=st->codec->width;
	(*frameYUV)->height=st->codec->height;
	(*frameYUV)->format=AV_PIX_FMT_YUV420P;

	// Alloc the images now
	sB1=avpicture_fill((AVPicture *)*frameRGB, buffRGB,AV_PIX_FMT_RGB24,(*frameRGB)->width, (*frameRGB)->height);
	sB2=avpicture_fill((AVPicture *)*frameYUV, buffYUV,AV_PIX_FMT_YUV420P,(*frameYUV)->width, (*frameYUV)->height);

	printf("Frames allocated with buffers size: %i, %i\n",sB1,sB2);
}
void deallocFrames(AVFrame *frameRGB, AVFrame *frameYUV, uint8_t *buffRGB, uint8_t *buffYUV){
	// Free
	free(buffRGB);
	free(buffYUV);

	// free allocated frame
	av_frame_free(&frameRGB);
	av_frame_free(&frameYUV);
}
void deallocFrameConversion(AVFrame *frameYUV,uint8_t *buffYUV){
	// Free
	free(buffYUV);

	// free allocated frame
	av_frame_free(&frameYUV);
}

// Movie header and allocation
int write_frame(AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt){
	/* rescale output packet timestamp values from codec to stream timebase */
	pkt->pts = av_rescale_q_rnd(pkt->pts, *time_base, st->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
	pkt->dts = av_rescale_q_rnd(pkt->dts, *time_base, st->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
	pkt->duration = av_rescale_q(pkt->duration, *time_base, st->time_base);
	pkt->stream_index = st->index;
    
	/* Write the compressed frame to the media file. */
	return av_interleaved_write_frame(fmt_ctx, pkt);
}
void writeHeader(const char *filename,AVFormatContext *oc){
	int ret;

	av_dump_format(oc,0,filename,1);

	ret = avio_open(&(oc->pb), filename, AVIO_FLAG_WRITE);
	if (ret < 0) {
		printf( "Could not open '%s.\n", filename);
		exit(1);
	}

	// write the header
	ret=avformat_write_header(oc,NULL);
	if (ret < 0){
		printf("Could not write the Header\n");
		exit(1);
	}
	printf("written\n");
}
void openFormat(const char * filename,AVFormatContext **oc){
	// Alloc the format context
	avformat_alloc_output_context2(oc,NULL,NULL,filename);
	if (!(*oc)){
		printf("Could not guess the output format: Continuing with mkv:\n");
		avformat_alloc_output_context2(oc,NULL,"mkv",filename);
	}
	// Suppress audio output
	(*oc)->oformat->audio_codec=CODEC_ID_NONE;
    
	// Check
	printf("Using %s: %s\nCodec: %s\n",(*oc)->oformat->name,(*oc)->oformat->long_name,avcodec_get_name((*oc)->oformat->video_codec));
}
void openStream(AVFormatContext *oc, AVCodec **codec, AVStream **st, int width, int height, int fps){
	// first find the codec
	*codec=avcodec_find_encoder(oc->oformat->video_codec);
	if (!(*codec)){
		printf("Codec not found\n");
		exit(1);
	}
	*st=avformat_new_stream(oc,*codec);
	if (!(*st)){
		printf("Could not allocate stream\n");
		exit(1);
	}
	(*st)->id=oc->nb_streams-1;
	av_opt_set((*st)->codec->priv_data,"preset","ultrafast",0);
	(*st)->codec->bit_rate = width*height*fps;
	(*st)->codec->bit_rate_tolerance = 2*(*st)->codec->bit_rate;
	(*st)->codec->codec_type=AVMEDIA_TYPE_VIDEO;
	/* resolution must be a multiple of two */
	(*st)->codec->width = width;
	(*st)->codec->height = height;
	/* frames per second */
	(*st)->codec->time_base.den = fps;
	(*st)->codec->time_base.num = 1;
	(*st)->codec->ticks_per_frame=2;
	//		(*st)->codec->gop_size = fps/2; /* emit one intra frame every ten frames */
	//		(*st)->codec->max_b_frames=5;
	(*st)->codec->refs = 4;
	(*st)->codec->gop_size = 4;
	(*st)->codec->max_b_frames = 4;
	(*st)->codec->me_range = 16;
	(*st)->codec->max_qdiff = 4;
	(*st)->codec->qmin = 15;
	(*st)->codec->qmax = 30;
	(*st)->codec->qcompress = 0.6;
	(*st)->codec->pix_fmt = AV_PIX_FMT_YUV420P;
	if (oc->oformat->flags & AVFMT_GLOBALHEADER){
		(*st)->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
	}
}
void openCodec(AVCodec **codec, AVStream *st){
	int ret;
	//open the codec
	ret=avcodec_open2(st->codec,*codec,NULL);
	if (ret <0){
		printf("Couldn't open the codec\n");
		exit(1);
	}
}

// Image processing
void rescaleRGBToYUV(AVFrame *framergb, AVFrame *frameyuv, uint8_t *buffrgb, uint8_t *buffyuv, size_t sizeB1){
	int ret;
	struct SwsContext *swsCtx;
    
	swsCtx = sws_getContext(framergb->width,framergb->height,AV_PIX_FMT_RGB24,frameyuv->width,frameyuv->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, 0,0,0);
	ret = sws_scale(swsCtx,framergb->data, framergb->linesize,0, framergb->height,frameyuv->data, frameyuv->linesize);
	printf("Image rescaled: %i\n",ret);
	sws_freeContext(swsCtx);
}
void rescaleYUV(AVFrame *frameyuv0, AVFrame *frameyuv1, uint8_t *buffrgb0, uint8_t *buffyuv1, size_t sizeB1){
	int ret;
	struct SwsContext *swsCtx;

	swsCtx = sws_getContext(frameyuv0->width,frameyuv0->height,AV_PIX_FMT_YUV420P,frameyuv1->width,frameyuv1->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, 0,0,0);
	ret = sws_scale(swsCtx,frameyuv0->data, frameyuv0->linesize,0, frameyuv0->height,frameyuv1->data, frameyuv1->linesize);
	printf("Image rescaled: %i\n",ret);
	sws_freeContext(swsCtx);
}
void encodeOneFrameYUV(AVFormatContext *oc,AVStream *st,AVFrame *frmyuv, int i){
	int ret,gotOutput;
    
	/* encode the image */
	AVPacket packet;
	av_init_packet(&packet);
	packet.data = NULL;
	packet.size = 0;
	fflush(stdout);
    
	frmyuv->pts=i;
	ret = avcodec_encode_video2(st->codec, &packet,frmyuv, &gotOutput);
	if (ret < 0) {
		printf("error encoding frame\n");
		exit(1);
	}
    
	if (gotOutput) {
		//			printf("encoding frame %3d (size=%5d)\n", itt+ifram, packet.size);
		ret=write_frame(oc,&st->codec->time_base, st,&packet);
	}
	av_free_packet(&packet);
}
