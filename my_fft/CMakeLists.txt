cuda_add_executable(my_fft myfft.cu)
cuda_add_cufft_to_target(my_fft)

file(COPY in.file DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

