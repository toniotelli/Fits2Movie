/****************************************/
/				Fits2Movie				 /
/****************************************/

This project is simply converting a files
series of Fits files into a movie using
The CUDA enable device.

The color table is based on the aia_lct.pro
from the SolarSoft distributed at:
http://www.lmsal.com/solarsoft.

In order to compile and run the project, 
you will need to have these libraries 
installed:

1. cfitsio: http://heasarc.gsfc.nasa.gov/docs/software/fitsio/fitsio.html
2. cuda: https://developer.nvidia.com/cuda-downloads
3. ffmpeg: http://ffmpeg.org/download.html

I have compiled it on Mac OS X Mavericks and
Ubuntu 14.04. The makefile depends on which 
OS you are running. I do not have a window
machine, I don't know how it will compile
under any version of window.

If you are under Mac OS, you need to define
CFITSIO_PATH pointing to the install directory 
of your cfitsio build.
If you are under Linux, you need that cfitsio.pc
is part of PKG_CONFIG_PATH

Building ...

In a terminal, type:

make clean all


Running ...

In a terminal, type:

./Fits2Movie

it will print a usage message

If you encounter any trouble, do not hesitate to contact me:
antoine.genetelli@mac.com
