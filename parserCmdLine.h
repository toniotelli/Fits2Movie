/*
 * parserCmdLine.h
 *
 *  Created on: May 29, 2014
 *      Author: tonio
 */

#ifndef PARSERCMDLINE_H_
#define PARSERCMDLINE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__

struct arguments{
	char *output;
	int itStart;
	double dMinMax[2];
	int NXNY[2];
	int scale;
	int fps;
};

static const char optString[]="d:s::f::";
int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments);

#else
#include <argp.h>
#include <argz.h>

/* Program documentation. */
static char doc[] = "Fits2Movie -- Convert sdo fits file to movie.";
static char args_doc[]="Output.mkv *.fits";

/* Programme option */
static struct argp_option options[]={
		{"scale",'d',"dmin:dmax",0,"Data value for data scaling"},
		{"size",'s',"nx:ny",OPTION_ARG_OPTIONAL,"resize YUV image to used Defined"},
		{"fps",'f',"FPS",OPTION_ARG_OPTIONAL,"User defined fps (Frame per second"},
		{0}
};

struct arguments{
	char *output;
	int itStart;
	double dMinMax[2];
	int NXNY[2];
	int scale;
	int fps;
};

error_t parse_opt (int key, char *arg, struct argp_state *state);
static struct argp argp = { options, parse_opt, args_doc, doc };
#endif


#endif /* PARSERCMDLINE_H_ */
