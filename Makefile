dirs :=  custom_fits easy_fits harder_fits my_fft test_fit

.PHONY: $(dirs) all clean
all: $(dirs)

clean: target=clean
clean: $(dirs)

$(dirs): | GooPdfCUDA.o
	+make -C $@ $(target)

include common.mk

GooPdfCUDA.o: $(WRKDIR)/CUDAglob.cu
	@echo "Compiling GooFit PDFs."
	$(NVCC) $(INCLUDES) $(CXXFLAGS) -DDUMMY=dummy -dc -o $@ $<
