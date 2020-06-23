# Configure these to your system
CUDA_SDK_PATH     = /opt/cudasdk
CUDA_TOOLKIT_PATH = /usr/local/cuda

ifeq ($(shell uname -m), x86_64)
ARCH	          = x86_64
GLARCH            = _x86_64
else
ARCH              = i386
GLARCH            =
endif

# Ordinary GCC/G++ settings
CXX               = g++
CFLAGS            = -g -Wall
LD                = g++
LDFLAGS           = -lGLEW$(GLARCH) -lglut -lGLU -lm 

VERSION           = 1.1
PKGNAME           = fractal-$(VERSION)

# Cuda and NVCC settings
NVCC              = nvcc
NVCCFLAGS         = -g -G
CUDA_INCLUDES     = -I$(CUDA_SDK_PATH)/C/common/inc
CUDA_LIBS         = -L $(CUDA_TOOLKIT_PATH)/lib64 -L $(CUDA_TOOLKIT_PATH)/lib \
	                -lcudart -L $(CUDA_SDK_PATH)/C/lib -L $(CUDA_SDK_PATH)/shared/lib/linux -lcutil_$(ARCH)
CUDA_SOURCES      = fractal.cu cudagl.cu gui.cu fractal-mandelbrot.cu fractal-julia.cu
X86_SOURCES       = glutils.cc

# Linking and target flags
LDFLAGS          += $(CUDA_LIBS)
OBJECTS           = $(X86_SOURCES:.cc=.o)    \
	                $(CUDA_SOURCES:.cu=.o)

all: fractal


# Default rule
%.o: %.cc
	$(CXX) $(CUDA_INCLUDES) $(CFLAGS) -c $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDES) -c $<

# Targets
fractal: $(OBJECTS)
	$(LD) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f *.o *.gz fractal
	
package:
	rm -f $(PKGNAME)
	ln -s . $(PKGNAME)
	tar czvf $(PKGNAME).tar.gz $(PKGNAME)/*.hh $(PKGNAME)/*.cc $(PKGNAME)/*.cu $(PKGNAME)/Makefile  
	rm -f $(PKGNAME)

