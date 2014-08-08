#!/bin/sh

if $1 eq 1; then 
	make clean -j 2 all
elif $1 eq 2; then
	Fits2Movie -d 0:1000 testRange.mkv /media/DD1/TestFits2movie/171/aia.lev1.171A_2010-06-11T00_*.fits
elif $1 eq 3; then
	Fits2Movie -s 1024:1024 testResize.mkv /media/DD1/TestFits2movie/171/aia.lev1.171A_2010-06-11T00_*.fits
elif $1 eq 4; then
	Fits2Movie -f 5 testFps.mkv /media/DD1/TestFits2movie/171/aia.lev1*.fits
fi

Fits2Movie -d 0:1000 -s 1024:1024 -f 10 testFull.mkv /media/DD1/TestFits2movie/171/aia.lev1.171A_2010-06-11T00_*.fits
Fits2Movie testNoArg.mkv /media/DD1/TestFits2movie/171/aia.lev1.171A_2010-06-11T00_*.fits