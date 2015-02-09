include ../common.mk
PDFOBJS := $(PDFS:.cu=.o)
all: $(PROGRAMS)

## Budowa własnej biblioteki PDFów zawierającej wszystkie istniejące
ifeq ($(TARGET_OMP),)
# Specjalne procedury dla CUDA
GooCUDA.o: ../GooPdfCUDA.o $(PDFOBJS)
	@echo "Linking all PDFs"
	$(NVCC) $(CXXFLAGS) -dlink $^ -o $@ 2>/dev/null


%Pdf.o: %Pdf.cu
	@echo "Compiling $<"
	$(NVCC) $(INCLUDES) $(CXXFLAGS) -dc -o $@ $<

# Linkowanie aplikacji
ifeq ($(PDFS),)
%: %.o
	@echo "Linking $^"
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(GOOBJS) $(WRKDIR)/GooPdfCUDA.o $< -o $@
else
%: %.o GooCUDA.o
	@echo "Linking $^ $(PDFOBJS)"
	$(LD) GooCUDA.o $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(GOOBJS) ../GooPdfCUDA.o $(PDFOBJS) $< -o $@
endif

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
