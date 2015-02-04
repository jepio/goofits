dirs :=  custom_fits easy_fits harder_fits my_fft test_fit

.PHONY: $(dirs) all clean
all: $(dirs)

clean: target=clean
clean: $(dirs)

$(dirs): 
	+make -C $@ $(target)
