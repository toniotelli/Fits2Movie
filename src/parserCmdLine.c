//
//  parserCmdLine.c
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//	email : antoine.genetelli@mac.com

#include "parserCmdLine.h"

// Printing Funciton
void printUsage(char *string){
	printf("Usage: %s [Options] filename.mkv *.fits\n",string);
}
void printHelp(char *string){
	// Welcome
	printf("\\************* %s *************\\\n",string);
	printf("%s is a small tool to convert fits images\nto any video format using CUDA to create rgb images.\n",string);
	printf("Written by Antoine Genetelli\nEmail: antoine.genetelli@mac.com\n\n");
	printUsage(string);
	// 
	printf("\nlist of avaible options:\n");
	printf("h --------- print this help\n");
	printf("d [min:max] set cut off limit for byte scaling images.\n");
	printf("s [nx:ny]   set new width and height to rescale the movie.\n");
	printf("f [fps]     set the desired frame per second rate.\n");
}
void printHead(char *message, int nx){
	int i=0;
	int	len=strlen(message)+4;

	// Compute the left indent
	int indent=(nx-len)/2;

	printf("\n\033[34m\\");
	for (i=0; i<indent; i++) printf("*");
	printf(" %s ",message);
	for (i=indent+len; i<nx; i++) printf("*");
	printf("/\033[0m\n");
}
void printProgress(int it, int itmax, int nx){
	char start[15],end[11];
	int indent, newNX, i;

	// Compute percentage done:
	float percPrg=it*100.0/(float)itmax;
	sprintf(start,"It = %i [",it);
	sprintf(end,"] %.3f%%",percPrg);
	printf("%s",start);

	// Compute indent and newNX
	newNX=nx-strlen(end)-strlen(start);
	indent=floor(percPrg*newNX/100.0);
	for (i=0; i< indent-1; i++) printf("=");
	if (i < newNX-1) printf(">");
	for (i=indent+1; i<newNX-1; i++) printf(" ");
	printf(" %s\n",end);
}

// Checking Function
bool checkArg(const char *filename){
	const char *ext=strchr(filename,'.');
	printf("ext = %s\n",ext);

	if (!ext) {
		return false;
	} else {
		if (strlen(ext) == strlen(".fits")){
			if (strcmp(ext,".fits")){
				printf("false");
				return false;
			} else {
				printf("true");
				return true;
			}
		} else if (strlen(ext) == strlen(".fts")) {
			if (!strcmp(ext,".fts")){
				printf("false");
				return false;
			} else {
				printf("true");
				return true;
			}
		} else {
			return true;
		}
	}
}
bool check2pow(unsigned int x){
	return ((x % 16) == 0);
}
// bool check2pow(unsigned int x){
// 	return ( x !=0 && !(x & x-1));
// }

// Image size checking -> Need to be a power of 2 for encoding
int recomputeImgSz(int x){
	// Get the lowest power of 2
	// float n=log((float)x)/log(2);
	// return (int)powf(2,floor(n)+1);
	return ((x/16)+1)*16;
}
bool checkImgSize(int *s0, int *s1){
	bool nX=check2pow(s0[0]);
	bool nY=check2pow(s0[1]);

	if (!nX || !nY) printf("\033[1;33mWarning :\033[0m Need to Resize Fits:\n");
	if (!nX) {
		s1[0]=recomputeImgSz(s0[0]);
		printf("----Change NX: %i to %i\n",s0[0],s1[0]);
	} else {
		s1[0]=s0[0];	
	}
	if (!nY) {
		s1[1]=recomputeImgSz(s0[1]);
		printf("----Change NY: %i to %i\n",s0[1],s1[1]);
	} else {
		s1[1]=s0[1];	
	}
	return (!nX || !nY);
}
bool checkUserSize(struct arguments *arguments){
		bool nX,nY;
		// check User value againt power of 2
		nX=check2pow(arguments->NX);
		nY=check2pow(arguments->NY);
		if (!nX || !nY) {
			printf("\033[31mResize need to be a power of 2!!!\033[0m\n");
			if (!nX) printf("----Change NX: %i to %i\n",arguments->NX,recomputeImgSz(arguments->NX));
			if (!nY) printf("----Change NY: %i to %i\n",arguments->NY,recomputeImgSz(arguments->NY));
			return 1;
		} else {
			return 0;
		}
}

// Determine output size
void getFinalSize(struct arguments *argm, int *outS, int *imgS, int *padS, int *resS){
	outS[0]=imgS[0];
	if (argm->resize){
		outS[1]=resS[1];
		outS[2]=resS[2];
	} else {
		if (argm->padding){
			outS[1]=padS[1];
			outS[2]=padS[2];
		} else {
			outS[1]=imgS[1];
			outS[2]=imgS[2];			
		}
	}
}

// Command line parser
int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments){
	int status = 0;
	int c=0;
	while (c != -1) {
		c=getopt(argc,argv,optString);
		switch(c){
			case 'd':
				arguments->scale=true;
				sscanf(optarg,"%lf:%lf",&(arguments->dMin),&(arguments->dMax));
				break;
			case 's':
				arguments->resize=true;
				sscanf(optarg,"%i:%i",&(arguments->NX),&(arguments->NY));
				if (checkUserSize(arguments)) return 1;
				break;
			case 'f':
				arguments->fpsU=true;
				sscanf(optarg,"%i",&(arguments->fps));
				break;
			case 'h':
				arguments->hFlag=true;
				printHelp(argv[0]);
				break;
			case '?':
				if (optopt == 'd' || optopt == 's' || optopt == 'f'){
					fprintf(stderr,"\033[31mOption %c requires an arguments.\033[0m\n",optopt);
					return 1;
				} else if (isprint(optopt)){
					fprintf(stderr,"\033[31mUnknown options -%c.\033[0m\n",optopt);
					return 1;
				} else {
					fprintf(stderr,"\033[31mUnknown option Character '\\x%x'.\033[0m\n",optopt);
					return 1;
				}
				break;
			// default:
			//      printUsage(argv[0]);
			//      return 1;
		}
	}

	if (arguments->hFlag) return 1;

	if (c == -1 && optind != argc-1) {
		arguments->argInd=optind;
		if (checkArg(argv[optind])){
			arguments->output=argv[optind];
			arguments->itStart=optind+1;
			return 0;
		} else {
			return 1;
		}
	} else {
		return c;
	}
}