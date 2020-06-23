# Configure these to your system
CUDA_SDK_PATH     = /opt/cudasdk
CUDA_TOOLKIT_PATH = /usr/local/cuda

ifeq ($(shell uname -m), x86_64)
ARCH	          = x86_64
else
ARCH              = i386
endif

# Ordinary GCC/G++ settings
CXX               = g++
CFLAGS            = -g -Wall
LD                = g++
LDFLAGS           = -lGLEW_x86_64 -lm -lglut

# Cuda and NVCC settings
NVCC              = nvcc
NVCCFLAGS         = -g
CUDA_INCLUDES     = -I$(CUDA_SDK_PATH)/C/common/inc
CUDA_LIBS         = -L $(CUDA_TOOLKIT_PATH)/lib64 -L $(CUDA_TOOLKIT_PATH)/lib \
	                -lcudart -L $(CUDA_SDK_PATH)/C/lib -L $(CUDA_SDK_PATH)/shared/lib/linux -lcutil_$(ARCH)
CUDA_SOURCES      = raytracer.cu

# Linking and target flags
LDFLAGS          += $(CUDA_LIBS)
OBJECTS           = $(X86_SOURCES:.cc=.o)    \
	                $(CUDA_SOURCES:.cu=.o)

all: main


# Default rule
%.o: %.cc
	$(CXX) $(CUDA_INCLUDES) $(CFLAGS) -c $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDES) -c $<

# Targets
main: $(OBJECTS) main.o
	$(LD) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f *.o main

