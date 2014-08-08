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
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <ctype.h>

struct arguments{
	char *output;
	bool hFlag;
	bool resize;
	bool scale;
	bool fpsU;
	int itStart;
	int fps;
	int NX;
	int NY;
	double dMin;
	double dMax;
	int argInd;
};

void printUsage(char *string);
void printHelp(char *string);
void printHead(char *message, int nx);
void printProgress(int it, int itmax, int nx);

bool checkArg(const char *filename);

const char optString[]="d:s:f:h";
int parseCmdLine(int argc, char *argv[], const char *optString, struct arguments *arguments);

#endif /* PARSERCMDLINE_H_ */
