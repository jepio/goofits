#ifndef MYFFT_H
#define MYFFT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "utils.h"

// Thrust
#ifdef THRUST

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

#endif


template <typename T> void read_data(std::vector<T> &data, std::string filename);

template <typename T> void fft_cuda(std::vector<T> in, std::vector<T> &out);

__global__ void magnitude(cufftComplex *, float * , size_t);

#endif /* MYFFT_H */
