/* 
 * Test of the CuFFT library. Calculating the absolute values of the complex
 * output can be done using a kernel or a thrust::transform operation.
 */

#include "myfft.h"

template <typename T> std::vector<T> read_data(std::string filename)
{
    std::vector<T> data;
    std::ifstream file;
    file.open(filename.c_str());
    T t, y;
    int i=0;
    while (file.good()){
        file >> t >> y;
        data.push_back(y);
        i++;
    }
    // Last element gets read twice this way. Remove it.
    data.pop_back();
    i--;
    std::cout << i <<std::endl; 
    file.close();
    return data;
}

__global__ void magnitude(cufftComplex *in, float *out, size_t size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size){
        out[idx] = cuCabsf(in[idx]);
    }
}

#ifdef THRUST
struct complex_mag_functor : public thrust::unary_function<cufftComplex, float>
{
    complex_mag_functor(){}

    __host__ __device__ float operator()(cufftComplex in)
    {
        return cuCabsf(in);
    }
};
#endif

template <typename T> std::vector<T> fft_cuda(std::vector<T>& in)
{
    cufftReal *d_in;
    size_t output_size = in.size()/2+1;
    // Copy input data to GPU
    checkCudaErrors(cudaMalloc((void **)&d_in,sizeof(T)*in.size()));
    checkCudaErrors(cudaMemcpy(d_in, &in[0], sizeof(T)*in.size(),cudaMemcpyHostToDevice));
    // Allocate space for output on GPU
    cufftComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out,sizeof(*d_out)*output_size));
    // Perform FFT
    cufftHandle plan;
    cufftPlan1d(&plan,in.size(), CUFFT_R2C,1); 
    cufftExecR2C(plan,d_in,d_out);
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
                    sizeof(*d_abs)*output_size,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    #endif
    // Thrust version
    #ifdef THRUST
    thrust::device_ptr<cufftComplex> dev_thr_out(d_out);
    out.resize(output_size);
    thrust::device_vector<T> thr_out(output_size);
    thrust::transform(dev_thr_out, dev_thr_out+output_size, thr_out.begin(),complex_mag_functor());
    thrust::copy(thr_out.begin(),thr_out.end(),&out[0]);
    cufftDestroy(plan);
    #endif
    cudaFree(d_in);
    cudaFree(d_out);
    #ifndef THRUST
    cudaFree(d_abs);
    #endif
    return out;
}

int main(void)
{
    std::vector<float> in;
    /*
     * Theoretically I used templates, however it will only work for floats.
     * Could be expanded to support doubles, but thats about all that can be
     * done.
     */
    in = read_data<float>("in.file");
    assert(in.size() != 0);
    std::vector<float> out;
    out = fft_cuda(in);
    std::ofstream outfile;
    outfile.open("fft.file");
    if (outfile.is_open()){
        for(unsigned int i=0;i<out.size();i++)
            outfile<<out[i]<<std::endl;
    }
    outfile.close();
    return 0;
}
