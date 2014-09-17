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

template <typename T> struct Vect;
template <> struct Vect<float>{
    typedef cuComplex type;
}
template <> struct Vect<double>{
    typedef cuDoubleComplex type;
}


template <typename T>
__device__ __host__ __inline__ T par_abs(Vect<T>::type in);
template <>
__device__ __host__ __inline__ float par_abs<float>(Vect<float>::type in){
    return cuCabsf(in);
}
template <>
__device__ __host__ __inline__ double par_abs<double>(Vect<double>::type in){
    return cuCabs(in);
}


template <typename T>
__global__ void magnitude(Vect<T>::type *, T * , size_t);

#endif /* MYFFT_H */
