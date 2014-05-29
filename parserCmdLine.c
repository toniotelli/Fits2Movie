/*
 * parserCmdLine.c
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

#include "parserCmdLine.h"

error_t parse_opt (int key, char *arg, struct argp_state *state){
	struct arguments *arguments=state->input;

	switch(key){
	case 'd':
		sscanf(arg,"=%lf:%lf",&(arguments->dMinMax[0]),&(arguments->dMinMax[1]));
		break;
	case 's':
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

