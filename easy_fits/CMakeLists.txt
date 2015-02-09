set(execs 
    sumfit
    prodfit
    gaussfit
    stepfit)

foreach(exec ${execs})
    cuda_add_executable(${exec} ${exec}.cu ${PDF_LIB})
    link_goofit(${exec})
endforeach()

file(GLOB texts *.txt)
file(COPY ${texts} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
