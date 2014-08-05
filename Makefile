################################################################################
#
# Build script for project
#
################################################################################
project_name := Fits2Movie

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
ifneq ($(DARWIN),)
	GCC             ?= gcc
	NVCC            ?= $(CUDA_BIN_PATH)/nvcc
else
	GCC             ?= clang
	NVCC            ?= $(CUDA_BIN_PATH)/nvcc -ccbin $(GCC)
endif

# Extra user flags
EXTRA_NVCCFLAGS ?= 
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?=


# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM35)

# Define Include and library PATH
PROJECT_PATH   = $(shell pwd)

PROJ_INCLUDES := -I/usr/local/cfitsio/include -I/usr/local/include -Isrc
PROJ_LIB      := -L. -L/usr/local/cfitsio/lib -L/usr/local/lib


# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath -Xlinker $(CUDA_LIB_PATH) $(PROJ_LIB) -lcfitsio -lavcodec -lavformat -lavutil -lswscale -lswresample
      CCFLAGS   := -Xcompiler -arch -Xcompiler $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -L/usr/local/cfitsio/lib $(PROJ_LIB) -lcuda -lcfitsio -lavcodec -lavformat -lavutil -lswscale -lswresample
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -L/usr/local/cfitsio/lib $(PROJ_LIB) -lcuda -lcfitsio -lavcodec -lavformat -lavutil -lswscale -lswresample
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 $(CCFLAGS)
else
      NVCCFLAGS := -m64 $(CCFLAGS)
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
#      NVCCFLAGS += -lineinfo
      TARGET    := release

endif

# Test if build folder presenet
ifneq ($(wildcard build),)
	mkdir build
	echo "create build dir"
endif

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. $(PROJ_INCLUDES)

# Source and object
SRCS := src/main.cu src/parserCmdLine.c src/aviFunction.c src/fitsFunction.c src/kernelConv.cu
OBJS := build/main.o build/parserCmdLine.o build/aviFunction.o build/fitsFunction.o build/kernelConv.o

# Target rules
all: $(project_name)

build: $(project_name)

build/%.o : src/%.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

build/%.o : src/%.c
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

$(project_name): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(EXTRA_LDFLAGS)

run:
	./$(project_name)

clean:
	rm -f $(project_name) build/*
    
printObjs: 
	echo $(OBJS)
printSrc: 
	echo $(SRCS)

