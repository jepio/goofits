cuda_add_executable(mixedfit mixedfit.cu ${PDF_LIB})
link_goofit(mixedfit)
file(COPY mixed.dat DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
