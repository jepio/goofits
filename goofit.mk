NVCC := nvcc
LD := g++
CXXFLAGS := -O3 -arch=sm_20

# Ustawienie potrzebne na Zeusie
ifneq ($(GOOFIT_ROOT),)
	GOOFITDIR := $(GOOFIT_ROOT)
endif

# Nagłówki CUDA oraz GooFit
ifeq ($(CUDALOCATION),)
	CUDALOCATION := $(GOOFITDIR)/fakecuda/
endif

# Nagłówki CUDA oraz GooFit
INCLUDES := -I$(CUDALOCATION)/include/ -I$(GOOFITDIR) -I$(GOOFITDIR)/rootstuff -I$(GOOFITDIR)/PDFs

# Nagłówki oraz biblioteki ROOT
ROOT_INCLUDES := -I$(shell root-config --incdir)
ROOT_LIBS := $(shell root-config --glibs)

## Przy kompilacji nvcc musi być tak:
## ROOT_LIBS = -L/software/local/tools/Root/5.34.09/lib/root -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread  -lm -ldl

# Biblioteki CUDA oraz GOOFIT
ifneq ($(TARGET_OMP),)
	CXXFLAGS += -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
	LDFLAGS += -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
endif

# Biblioteki CUDA oraz specjalna ROOT
LIBS := -L$(CUDALOCATION)/lib64 -L$(GOOFITDIR)/rootstuff -lRootUtils
ifeq ($(TARGET_OMP),)
	LIBS += -lcudart
endif

LIBS += $(ROOT_LIBS)

# Obiekty GooFit
WRKDIR := $(GOOFITDIR)/wrkdir/
GOOBJS := $(WRKDIR)/Variable.o $(WRKDIR)/FitManager.o $(WRKDIR)/Faddeeva.o $(WRKDIR)/FitControl.o $(WRKDIR)/PdfBase.o $(WRKDIR)/DataSet.o $(WRKDIR)/BinnedDataSet.o $(WRKDIR)/UnbinnedDataSet.o $(WRKDIR)/FunctorWriter.o

PDFOBJS := $(PDFS:.cu=.o)
all: $(PROGRAMS)

## Budowa własnej biblioteki PDFów zawierającej wszystkie istniejące
ifeq ($(TARGET_OMP),)
# Specjalne procedury dla CUDA
GooCUDA.o: GooPdfCUDA.o $(PDFOBJS)
	@echo "Linking all PDFs"
	$(NVCC) $(CXXFLAGS) -dlink $^ -o $@ 2>/dev/null

GooPdfCUDA.o: $(WRKDIR)/CUDAglob.cu
	@echo "Compiling GooFit PDFs."
	$(NVCC) $(INCLUDES) $(CXXFLAGS) -DDUMMY=dummy -dc -o $@ $<

%Pdf.o: %Pdf.cu
	@echo "Compiling $<"
	$(NVCC) $(INCLUDES) $(CXXFLAGS) -dc -o $@ $<

# Linkowanie aplikacji
%: %.o GooCUDA.o
	@echo "Linking $^ $(PDFOBJS)"
	$(LD) GooCUDA.o $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(GOOBJS) GooPdfCUDA.o $(PDFOBJS) $< -o $@

else
# Dla OMP nie sa wymagane zadne specjalne reguly
%: $(PDFOBJS) %.o
	@echo "Linking $^"
	$(LD) $(LDFLAGS) $(LIBS) $(WRKDIR)/*.o $^ -o $@

endif

# Kompilacja aplikacji
%.o: %.cu
	@echo "Compiling $<"
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(ROOT_INCLUDES) -c -o $@ $<

clean:
	rm -f $(PDFOBJS) $(PROGRAMS)
