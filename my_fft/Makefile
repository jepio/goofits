ifneq ($(THRUST),)
	CPPFLAGS+=-DTHRUST
endif

all: myfft

myfft: myfft.cu
	nvcc $(CPPFLAGS) -Xcompiler -Wall -arch=sm_20 -O2 -lcufft -o myfft myfft.cu

clean:
	rm -rf myfft
