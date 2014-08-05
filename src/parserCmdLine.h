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
#include <stdbool.h>
#include <unistd.h>
#include <ctype.h>

typedef struct {
	char *output;
	bool hFlag;
	bool resize;
	bool scale;
	bool fpsU;
	int itStart;
	double dMinMax[2];
	int NXNY[2];
	int fps;
}arguments;

void printUsage(char *string);
void printHelp(char *string);
static const char optString[]="d:s::f::h";
int parseCmdLine(int argc, char *argv[], const char *optString, arguments *arguments);

#endif /* PARSERCMDLINE_H_ */
