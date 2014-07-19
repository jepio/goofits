CXX=nvcc
LD=g++
CXXFLAGS=-O3 -arch=sm_20

# Ustawienie potrzebne na Zeusie
ifneq ($(GOOFIT_ROOT),)
GOOFITDIR=$(GOOFIT_ROOT)
endif

# Nagłówki CUDA oraz GooFit
ifeq ($(CUDALOCATION),)
CUDALOCATION=$(GOOFITDIR)/fakecuda/
endif

INCLUDES = -I$(CUDALOCATION)/include/ -I$(GOOFITDIR) -I$(GOOFITDIR)/rootstuff -I$(GOOFITDIR)/PDFs

# Nagłówki oraz biblioteki ROOT
ROOT_INCLUDES= -I$(shell root-config --incdir)
ROOT_LIBS = $(shell root-config --glibs)

# Biblioteki CUDA oraz GOOFIT
ifneq ($(TARGET_OMP),)
CXXFLAGS += -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
LDFLAGS += -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
endif

LIBS = -L$(CUDALOCATION)/lib64 -L$(GOOFITDIR)/rootstuff -lRootUtils
ifeq ($(TARGET_OMP),)
LIBS += -lcudart
endif

WRKDIR=$(GOOFITDIR)/wrkdir/

# Nazwa własnej aplikacji
PROGRAMS=sumfit prodfit gaussfit stepfit

all: $(PROGRAMS)

# Linkowanie aplikacji
%: %.o
	$(LD) $(LDFLAGS) $(LIBS) $(ROOT_LIBS) $(WRKDIR)/*.o $^ -o $@
	@echo "Building $* is done"

# Kompilacja aplikacji
%.o: %.cu
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ROOT_INCLUDES) -c -o $@ $<


clean:
	@rm -f *.o $(PROGRAMS)
