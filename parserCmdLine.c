/*
 * parserCmdLine.c
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

#include "parserCmdLine.h"

#ifdef __APPLE__

int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments){
	int c=0;
	opterr = 0;
	while ((c=getopt(argc,argv,"d:s::")) != -1)
		switch(c){
		case 'd':
			printf("option d: %s ",optarg);
			sscanf(optarg,"=%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
			printf("dmin=%lf, dmax= %lf\n",arguments->dMinMax[0],arguments->dMinMax[1]);
			break;
		case 's':
			arguments->scale=1;
			sscanf(optarg,"=%i:%i",&(arguments->NXNY[0]),&(arguments->NXNY[1]));
			break;
		case '?':
			if (optopt == 'd') fprintf(stderr,"Option %c requires an arguments.\n",optopt);
			return 1;
		default:
			abort();
		}

	arguments->output=argv[optind];
	arguments->itStart=optind+1;
	return 0;
}

#else
error_t parse_opt (int key, char *arg, struct argp_state *state){
	struct arguments *arguments=state->input;

	switch(key){
	case 'd':
		sscanf(arg,"=%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
		break;
	case 's':
		arguments->scale=1;
		sscanf(arg,"=%i:%i",&(arguments->NXNY[0]),&(arguments->NXNY[1]));
		break;
	case ARGP_KEY_ARG:
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
