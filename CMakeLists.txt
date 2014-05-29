CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(Fits2Movie)


set(source_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(binary_dir ${CMAKE_CURRENT_BINARY_DIR})

set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_C_LINK_EXECUTABLE /usr/bin/gcc)

#find cuda
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message("cuda_path ${CUDA_INCLUDE_DIRS}")
  message("cuda_path ${CUDA_LIBRARY_DIRS}")
  message("cuda_path ${CUDA_LIBRARIES}")
  set(cuda_path ${CUDA_TOOLKIT_ROOT_DIR})
  set(CUDA_HOST_COMPILER /usr/bin/gcc) 
else(CUDA_FOUND)
   message(FATAL_ERROR "CUDA could not be found")
endif(CUDA_FOUND)

if(NOT CMAKE_HOST_APPLE)
	find_package(PkgConfig)
	if (PKG_CONFIG_FOUND)
		message("using pkg-config...")
		pkg_check_modules(AVCODEC REQUIRED libavcodec)
		pkg_check_modules(AVUTIL REQUIRED libavutil)
		pkg_check_modules(AVFORMAT REQUIRED libavformat)
		pkg_check_modules(SWSCALE REQUIRED libswscale)
		pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
		pkg_check_modules(CFITSIO REQUIRED cfitsio)
	endif(PKG_CONFIG_FOUND)
else(NOT CMAKE_HOST_APPLE)
	# Include dir
	set(AVCODEC_INCLUDE_DIRS "/usr/local/include")
	set(AVUTIL_INCLUDE_DIRS "/usr/local/include")
	set(AVFORMAT_INCLUDE_DIRS "/usr/local/include")
	set(SWSCALE_INCLUDE_DIRS "/usr/local/include")
	set(RESAMPLE_INCLUDE_DIRS "/usr/local/include")
	set(CFITSIO_INCLUDE_DIRS "$ENV{CFITSIO_PATH}/include")
	# Libraries dir
	set(AVCODEC_LIBRARY_DIRS "/usr/local/lib")
	set(AVUTIL_LIBRARY_DIRS "/usr/local/lib")
	set(AVFORMAT_LIBRARY_DIRS "/usr/local/lib")
	set(SWSCALE_LIBRARY_DIRS "/usr/local/lib")
	set(RESAMPLE_LIBRARY_DIRS "/usr/local/include")
	set(CFITSIO_LIBRARY_DIRS "$ENV{CFITSIO_PATH}/lib")
	# Target library
	set(AVCODEC_LIBRARIES "-lavcodec")
	set(AVUTIL_LIBRARIES "-lavutil")
	set(AVFORMAT_LIBRARIES "-lavformat")
	set(SWSCALE_LIBRARIES "-lswscale")
	set(RESAMPLE_LIBRARIES "-lswresample")
	set(CFITSIO_LIBRARIES "-lcfitsio")
endif(NOT CMAKE_HOST_APPLE)

# includes
include_directories(
${AVCODEC_INCLUDE_DIRS}
${AVUTIL_INCLUDE_DIRS}
${AVFORMAT_INCLUDE_DIRS}
${SWSCALE_INCLUDE_DIRS}
${SWRESAMPLE_INCLUDE_DIRS}
${CFITSIO_INCLUDE_DIRS}
${cuda_path}/include
${cuda_path}/samples/common/inc
${source_dir}
${binary_dir}
)

# PARAMETER THE LIBRARY
set(libsCuda "-L${cuda_path}/lib64 -lcuda")


# Set cuda compilation flags
set(custom_cuda_flags "-g -gencode arch=compute_35,code=sm_35")
set(CUDA_NVCC_FLAGS ${custom_cuda_flags})
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpermissive")

# Message
message("avcodec: ${AVCODEC_LIBRARIES}")

add_library(parserlib STATIC parserCmdLine.c parserCmdLine.h)
CUDA_ADD_EXECUTABLE(Fits2Movies main.cu kernelConv.cu kernelConv.cuh fitsFunction.c aviFunction.c fitsFunction.h aviFunction.h)
TARGET_LINK_LIBRARIES(Fits2Movies parserlib ${libsCuda} ${CFITSIO_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${SWSCALE_LIBRARIES} ${SWRESAMPLE_LIBRARIES})

#CUDA_ADD_EXECUTABLE(Fits2Movies main.cu)# kernelConv.cu kernelConv.cuh)# fitsFunction.c aviFunction.c fitsFunction.h aviFunction.h)
#cuda_add_library(cudaLib STATIC kernelConv.cu kernelConv.cuh)
#add_library(cpuFunc STATIC fitsFunction.c aviFunction.c fitsFunction.h aviFunction.h)
#TARGET_LINK_LIBRARIES(Fits2Movies cudaLib cpuFunc ${libsCuda} ${CFITSIO_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${SWSCALE_LIBRARIES} ${SWRESAMPLE_LIBRARIES})


