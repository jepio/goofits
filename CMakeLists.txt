cmake_minimum_required(VERSION 2.8)

project(goofits)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(ROOT REQUIRED)
find_package(CUDA REQUIRED)

### goofit Includes
set(GOOFIT $ENV{GOOFIT_ROOT})
if (GOOFIT STREQUAL "")
    message(FATAL "Goofit directory not found")
else()
    message(STATUS "Found Goofit directory: ${GOOFIT}")
    cuda_include_directories(${GOOFIT} ${GOOFIT}/rootstuff ${GOOFIT}/PDFs)
endif()
### root includes
cuda_include_directories(${ROOT_INCLUDE_DIR})
### cuda flags
list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
### link to goofit root lib
find_library(ROOTUTILS_LIB
    RootUtils PATHS ${GOOFIT}/rootstuff)
if(NOT ROOTUTILS_LIB)
    message(FATAL "Couldn't find libRootUtils.so")
endif()
### goofit objects
set(GOOOBJS 
    Variable.o FitManager.o Faddeeva.o FitControl.o PdfBase.o DataSet.o
    BinnedDataSet.o UnbinnedDataSet.o FunctorWriter.o)
unset(_tmp)
foreach(src ${GOOOBJS})
    list(APPEND _tmp ${GOOFIT}/wrkdir/${src})
endforeach()
set(GOOOBJS ${_tmp})
add_library(goofit ${GOOOBJS})
set_target_properties(goofit PROPERTIES LINKER_LANGUAGE CXX)
### build library of PDFs
cuda_add_library(pdf_lib SHARED ${GOOFIT}/wrkdir/CUDAglob.cu 
    OPTIONS -DDUMMY=dummy)
function(link_goofit TARGET)
    target_link_libraries(${TARGET} ${ROOTUTILS_LIB} pdf_lib goofit
        ${ROOT_LIBRARIES})
endfunction()

add_subdirectory(my_fft)
add_subdirectory(easy_fits)
add_subdirectory(test_fit)
add_subdirectory(harder_fits)
add_subdirectory(custom_fits)

