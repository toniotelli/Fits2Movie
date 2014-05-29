#include "fitsFunction.h"

size_t allocDataType(void **data,int datatype,int nx, int ny){
	size_t sizeData = 0;
	switch(datatype) {
	case BYTE_IMG:
		sizeData = nx*ny*sizeof(unsigned char);
		break;
	case SHORT_IMG:
		sizeData = nx*ny*sizeof(short int);
		break;
	case LONG_IMG:
		sizeData = nx*ny*sizeof(long);
		break;
	case FLOAT_IMG:
		sizeData = nx*ny*sizeof(float);
		break;
	case DOUBLE_IMG:
		sizeData = nx*ny*sizeof(double);
		break;
	}
	*data=malloc(sizeData);
	return sizeData;
}
int getImageSize(const char *filename,int *imgS, int *min, int *max){
	fitsfile *fts;
	int status=0;

	// open file
	fits_open_file(&fts,filename,READONLY,&status);
	if (status != 0 ){
		fits_report_error(stderr,status);
		exit(-1);
	}

	// Read bitpix, NAXIS1, NAXIS2
	fits_read_key(fts,TINT,"BITPIX",(void *)&imgS[0],NULL,&status);
	fits_read_key(fts,TINT,"NAXIS1",(void *)&imgS[1],NULL,&status);
	fits_read_key(fts,TINT,"NAXIS2",(void *)&imgS[2],NULL,&status);

	// Read DATAMIN, DATAMAX
	fits_read_key(fts,TINT,"DATAMIN",(void *)min,NULL,&status);
	fits_read_key(fts,TINT,"DATAMAX",(void *)max,NULL,&status);
	*min=*min-1;
	*max=*max+1;

	// close the file
	fits_close_file(fts,&status);
	return status;
}
int readFits(const char *filename,void *data, int *imgS, int *min, int *max){
	fitsfile *fts;
	int status=0,datatype=0;
	double nulval=0;

	// open file
	fits_open_file(&fts,filename,READONLY,&status);
	if (status != 0 ){
		fits_report_error(stderr,status);
		exit(-1);
	}

	// Read DATAMIN, DATAMAX
	fits_read_key(fts,TINT,"DATAMIN",(void *)min,NULL,&status);
	fits_read_key(fts,TINT,"DATAMAX",(void *)max,NULL,&status);
	*min=*min-1;
	*max=*max+1;
	printf("[min,max]=[%i,%i]\n",*min,*max);

	// Data type
	switch(imgS[0]) {
	case BYTE_IMG:
		datatype = TBYTE;
		break;
	case SHORT_IMG:
		datatype = TSHORT;
		break;
	case LONG_IMG:
		datatype = TLONG;
		break;
	case FLOAT_IMG:
		datatype = TFLOAT;
		break;
	case DOUBLE_IMG:
		datatype = TDOUBLE;
		break;
	}

	// Read the image
	fits_read_img(fts, datatype, 1, imgS[1]*imgS[2], (void *)&nulval, data, NULL, &status);
	if (status != 0 ){
		fits_report_error(stderr,status);
		exit(-1);
	}

	// close the file
	fits_close_file(fts,&status);

	return status;
}

