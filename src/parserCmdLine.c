/*
 * parserCmdLine.c
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

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
		} else {
			return true;
		}
	}
}

int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments){
	int status = 0;
	int c=0;
	while (c != -1) {
		// c=getopt(argc,argv,"d:s::f::h");
		c=getopt(argc,argv,optString);
		switch(c){
		case 'd':
			arguments->scale=true;
			sscanf(optarg,"%lf:%lf",&(arguments->dMin),&(arguments->dMax));
			break;
		case 's':
			arguments->resize=true;
			sscanf(optarg,"%i:%i",&(arguments->NX),&(arguments->NY));
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
		// 	printUsage(argv[0]);
		// 	return 1;
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
		return 0;
	}else {
		return c;
	}
}