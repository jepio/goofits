My GooFit Applications
======================

These are the applications I wrote to test the
[GooFit](https://github.com/GooFit/GooFit/) GPU fitting framework.
They compile on the supercomputer Zeus at ACK Cyfronet, compiling them elsewhere requires adjustments to the Makefiles. 

These have been prepared with the CUDA backend in mind, however nothing stands
in the way of building them for OMP.

To test them run `make` and then launch the binary. The data `.dat` files are expected
to be in the same directory as the executable. 
