//
//  fitsFunction.h
//  Fits2Movie
//
//  Created by Antoine Genetelli on 27/05/14.
//  Copyright (c) 2014 Antoine Genetelli. All rights reserved.
//

#ifndef Fits2Movie_fitsFunction_h
#define Fits2Movie_fitsFunction_h

#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>

size_t allocDataType(void **data,int datatype,int nx, int ny);
int getImageSize(const char *filename,int *imgS, int *min, int *max,int *wave);
int readFits(const char *filename,void *data, int *imgS, int *min, int *max);


#endif
