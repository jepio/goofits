CXX=nvcc
LD=g++
CXXFLAGS=-O3 -arch=sm_20

# Ustawienie potrzebne na Zeusie
GOOFITDIR=$(GOOFIT_ROOT)


# Nagłówki CUDA oraz GooFit
INCLUDES = -I$(CUDALOCATION)/include/ -I$(GOOFITDIR) -I$(GOOFITDIR)/rootstuff -I$(GOOFITDIR)/PDFs

# Nagłówki oraz biblioteki ROOT
ROOT_INCLUDES= -I$(shell root-config --incdir)
ROOT_LIBS = $(shell root-config --glibs)

# Biblioteki CUDA oraz GOOFIT
LIBS = -L$(CUDALOCATION)/lib64 -lcudart -L$(GOOFITDIR)/rootstuff -lRootUtils
WRKDIR=$(GOOFITDIR)/wrkdir/

# Nazwa własnej aplikacji
PROGRAM=mixedfit

all: $(PROGRAM)

# Linkowanie aplikacji
%: %.o
	$(LD) $(LIBS) $(ROOT_LIBS) $(WRKDIR)/*.o $^ -o $@ 
	@echo "Building $* is done"

# Kompilacja aplikacji
%.o: %.cu
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(ROOT_INCLUDES) -c -o $@ $<


clean:
	@rm -f *.o $(PROGRAM)
