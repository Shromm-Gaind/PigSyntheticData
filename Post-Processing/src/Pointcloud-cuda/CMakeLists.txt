cmake_minimum_required(VERSION 3.28)
project(CreatePointcloud LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA architectures to target
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Open3D
find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})


# Specify the path to the CUDA toolkit
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Include CUDA directories
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# Add executable
add_executable(CreatePointcloud main.cpp kernels.cu)

target_link_libraries(CreatePointcloud ${OpenCV_LIBS} ${Open3D_LIBRARIES})
