/*
 * Test of the CuFFT library. Calculating the absolute values of the complex
 * output can be done using a kernel or a thrust::transform operation.
 */

#include "myfft.h"
#include <stdexcept>

template <typename T> std::vector<T> read_data(std::string filename)
{
    std::vector<T> data;
    std::ifstream file(filename.c_str());
    if (!file)
        throw std::runtime_error("File couldn't be opened.");

    T t, y;
    while (file >> t >> y){
        data.push_back(y);
    }

    std::cout << data.size() <<std::endl;
    return data;
}

template <typename T>
__global__ void magnitude(typename Vect<T>::type *in, T *out, size_t size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size){
        out[idx] = par_abs(in[idx]);
    }
}

#ifdef THRUST
template <typename T>
struct complex_mag_functor : public thrust::unary_function<typename Vect<T>::type, T> {
    complex_mag_functor(){}

    __host__ __device__ T operator()(typename Vect<T>::type in)
    {
        return par_abs(in);
    }
};
#endif

template <typename T> std::vector<T> fft_cuda(const std::vector<T>& in)
{
    T *d_in;
    size_t output_size = in.size() / 2 + 1;
    // Copy input data to GPU
    checkCudaErrors(cudaMalloc((void **)&d_in, sizeof(T) * in.size()));
    checkCudaErrors(cudaMemcpy(d_in, &in[0], sizeof(T) * in.size(), cudaMemcpyHostToDevice));
    // Allocate space for output on GPU
    typename Vect<T>::type *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(*d_out) * output_size));
    // Perform FFT
    cufftHandle plan;
    cufftPlan1d(&plan, in.size(), Vect<T>::plantype, 1);
    Vect<T>::ptr(plan, d_in, d_out);
    // Calculate absolute values on GPU and copy to CPU
    std::vector<T> out;

#ifndef THRUST
    T *d_abs;
    checkCudaErrors(cudaMalloc((void **)&d_abs, sizeof(*d_abs)*output_size));
    size_t blockSize = 1024;
    size_t gridSize = output_size / blockSize + 1;
    magnitude <<<gridSize, blockSize>>>(d_out, d_abs, output_size);

    out.resize(output_size);
    checkCudaErrors(cudaMemcpy(&out[0], d_abs,
                    sizeof(*d_abs) * output_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(d_abs);
#endif
    // Thrust version
#ifdef THRUST
    thrust::device_ptr<typename Vect<T>::type> dev_thr_out(d_out);
    out.resize(output_size);
    thrust::device_vector<T> thr_out(output_size);
    thrust::transform(dev_thr_out, dev_thr_out + output_size, thr_out.begin(), complex_mag_functor<T>());
    thrust::copy(thr_out.begin(),thr_out.end(), &out[0]);
#endif
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

int main()
{
    // templated calculation type should be chosen here (convenience)
    typedef std::vector<float> data_v;

    data_v in = read_data<data_v::value_type>("in.file");

    assert(in.size() != 0);

    data_v out = fft_cuda(in);
    std::ofstream outfile("fft.file");

    if (outfile){
        for(data_v::iterator i = out.begin(), e = out.end(); i != e; ++i)
            outfile << *i << std::endl;
    }

    return 0;
}
