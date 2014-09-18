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
    static const cufftType plantype = CUFFT_R2C;
    typedef cufftResult (*func)(cufftHandle, cufftReal*,cufftComplex*); 
    static func ptr;
};
template <> struct Vect<double>{
    typedef cuDoubleComplex type;
    static const cufftType plantype = CUFFT_D2Z;
    typedef cufftResult (*func)(cufftHandle, cufftDoubleReal*,cufftDoubleComplex*);
    static func ptr;
};

Vect<float>::func Vect<float>::ptr = cufftExecR2C;
Vect<double>::func Vect<double>::ptr = cufftExecD2Z;

template <typename T>
__device__ __host__ __inline__ T par_abs(typename Vect<T>::type in);
template <>
__device__ __host__ __inline__ float par_abs<float>(Vect<float>::type in){
    return cuCabsf(in);
}
template <>
__device__ __host__ __inline__ double par_abs<double>(Vect<double>::type in){
    return cuCabs(in);
}


template <typename T>
__global__ void magnitude(typename Vect<T>::type *, T * , size_t);

#endif /* MYFFT_H */
