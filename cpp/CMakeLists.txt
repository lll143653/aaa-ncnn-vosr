cmake_minimum_required(VERSION 3.10)
project(vosr_sr CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ncnn SDK location (headers + prebuilt libs). Override on the command line:
#   cmake -S cpp -B cpp/build -DNCNN_INCLUDE_DIR=<ncnn>/include/ncnn -DNCNN_LIB_DIR=<ncnn>/lib
set(NCNN_INCLUDE_DIR "" CACHE STRING "ncnn include dir (contains net.h, mat.h)")
set(NCNN_LIB_DIR "" CACHE PATH "Path to directory containing ncnn.lib / libncnn.a")
if(NOT NCNN_INCLUDE_DIR OR NOT NCNN_LIB_DIR)
    message(FATAL_ERROR "ncnn not found. Re-run cmake with:\n"
            "  -DNCNN_INCLUDE_DIR=<ncnn install>/include/ncnn -DNCNN_LIB_DIR=<ncnn install>/lib")
endif()

# opencv-mobile under ../3rdparty (auto-detected)
file(GLOB OPENCV_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv-mobile-*/x64/lib/cmake/opencv4/OpenCVConfig.cmake"
                        "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv-mobile-*/x64/OpenCVConfig.cmake")
if(NOT OPENCV_CONFIG)
    message(FATAL_ERROR "opencv-mobile not found under 3rdparty/")
endif()
list(GET OPENCV_CONFIG 0 OPENCV_CONFIG)
get_filename_component(OpenCV_DIR "${OPENCV_CONFIG}" DIRECTORY)
set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)

# ncnn + glslang stack (required by ncnn's Vulkan support)
set(NCNN_LIBS ncnn glslang SPIRV MachineIndependent OSDependent GenericCodeGen glslang-default-resource-limits)

add_executable(vosr_sr src/vosr_sr.cpp)
target_include_directories(vosr_sr PRIVATE include ${NCNN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_directories(vosr_sr PRIVATE ${NCNN_LIB_DIR})
target_link_libraries(vosr_sr PRIVATE ${NCNN_LIBS} ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(vosr_sr PRIVATE /O2 /openmp /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(vosr_sr PRIVATE -O2 -fopenmp)
    target_link_libraries(vosr_sr PRIVATE -fopenmp)
endif()
