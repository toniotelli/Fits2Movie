/*
 * parserCmdLine.c
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

#include "parserCmdLine.h"

#ifdef __APPLE__

int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments){
	int status = 0;
	char *ext=(char *)malloc(6);
	int c=0;
	while (c != -1) {
		c=getopt(argc,argv,"d:s::f::");
		switch(c){
		case 'd':
			sscanf(optarg,"%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
			break;
		case 's':
			arguments->scale=1;
			sscanf(optarg,"%i:%i",&(arguments->NXNY[0]),&(arguments->NXNY[1]));
			break;
		case 'f':
			sscanf(optarg,"%i",&(arguments->fps));
			c=-1;
			break;
		case '?':
			if (argc == 1) {
				status=-1;
				printf("Usage: %s [-d] [min:max] [-s] [nx:ny] [-f] [fps] filename.mkv *.fits\n",argv[0]);
			} else if (optopt == 'd' || optopt == 's' || optopt == 'f'){
				fprintf(stderr,"Option %c requires an arguments.\n",optopt);
				status=-1;
			} else if (isprint(optopt)){
				fprintf(stderr,"Unknown options -%c.\n",optopt);
				status=-1;
			} else {
				fprintf(stderr,"Unknown option Character '\\x%x'.\n",optopt);
				status=-1;
			}
			return 1;
		default:
			abort();
		}
	}

	if (c == -1 && status != -1) {
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

#else
error_t parse_opt (int key, char *arg, struct argp_state *state){
	char *ext=(char *)malloc(6);
	struct arguments *arguments=state->input;
	switch(key){
	case 'd':
		printf("arg=%s\n",arg);
		sscanf(arg,"=%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
		break;
	case 's':
		printf("arg=%s\n",arg);
		arguments->scale=1;
		sscanf(arg,"=%i:%i",&(arguments->NXNY[0]),&(arguments->NXNY[1]));
		break;
	case 'f':
		printf("arg=%s\n",arg);
		sscanf(arg,"=%i",&(arguments->fps));
		break;
	case ARGP_KEY_ARG:
		printf("arg=%s\n",arg);
		arguments->output=arg;
		arguments->itStart=state->next;
		state->next=state->argc;
		break;
	case ARGP_KEY_NO_ARGS:
		argp_usage(state);
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}
#endif
