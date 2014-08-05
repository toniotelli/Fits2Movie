/*
 * parserCmdLine.c
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

#include "parserCmdLine.h"

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

int parseCmdLine(int argc, char *argv[], const char *optString, arguments *arguments){
	int status = 0;
	char *ext=(char *)malloc(6);
	int c=0;
	while (c != -1) {
		// c=getopt(argc,argv,"d:s::f::h");
		c=getopt(argc,argv,optString);
		switch(c){
		case 'd':
			arguments->scale=true;
			sscanf(optarg,"%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
			break;
		case 's':
			arguments->resize=true;
			sscanf(optarg,"%i:%i",&(arguments->NXNY[0]),&(arguments->NXNY[1]));
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
				fprintf(stderr,"Option %c requires an arguments.\n",optopt);
				return 1;
			} else if (isprint(optopt)){
				fprintf(stderr,"Unknown options -%c.\n",optopt);
				return 1;
			} else {
				fprintf(stderr,"Unknown option Character '\\x%x'.\n",optopt);
				return 1;
			}
			break;
		// default:
		// 	printUsage(argv[0]);
		// 	return 1;
		}
	}

	if (arguments->hFlag) return 1;

	if (c == -1) {
		int count = 0;
		for (int ind = optind; ind < argc; ind++){
			if (strlen(argv[ind]) > 6){
				strncpy(ext,argv[ind]+strlen(argv[ind])-5,strlen(argv[ind]));
				if ((strcmp(ext,".fits") != 0 )){
					if (*(argv[ind]) != '-'){
						arguments->output=argv[ind];
					}
				} else if (count == 0){
					arguments->itStart=ind;
					count++;
				}
			}
		}
		free(ext);
		return 0;
	}else {
		free(ext);
		return c;
	}
}