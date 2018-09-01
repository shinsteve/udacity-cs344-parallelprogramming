############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cs344)

# tell cmake where the FindXXX.cmake functions are
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV2_LIBS} )
include_directories(/usr/local/cuda/samples/common/inc)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(OPENCV2_FOUND)
    message( STATUS "OpenCV2 found" )
    message( STATUS ${OpenCV_LIBS} )
else(OPENCV2_FOUND)
    message( STATUS "OpenCV2 NOT found" )
endif(OPENCV2_FOUND)

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/clang; 
  -gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_35,code=compute_35;")
#  -gencode;arch=compute_20,code=sm_20;
#  -gencode;arch=compute_11,code=sm_11; 
#  -gencode;arch=compute_12,code=sm_12;
#  -gencode;arch=compute_13,code=sm_13;")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
#    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory (hw1)
  # add_subdirectory (hw2)
  # add_subdirectory (hw3)
  # add_subdirectory (hw4)
  # add_subdirectory (hw5)
  # add_subdirectory (hw6)
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
