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


template <typename T> std::vector<T> read_data(std::string filename);

template <typename T> std::vector<T> fft_cuda(std::vector<T>& in);

template <typename T>
struct vector_type{
    T x, y;
}

template <typename T>
__device__ __host__ __inline__ T par_abs(vector_type<T> in)
{
    return sqrt(in.x*in.x + in.y*in.y);
}

template <typename T>
__global__ void magnitude(vector_type<T> *, T * , size_t);

#endif /* MYFFT_H */
