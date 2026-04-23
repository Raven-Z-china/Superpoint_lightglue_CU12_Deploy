cmake_minimum_required(VERSION 3.5)
project(superpoint_lightglue)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_BUILD_TYPE "Release")
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)


find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT 10.8 paths
set(TENSORRT_ROOT ~/tensorrt/TensorRT-10.8.0.43)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_LIBS nvinfer nvonnxparser)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}_lib SHARED
        src/super_point.cpp
        src/light_glue.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        yaml-cpp
        tensorrtbuffer
)

# Add TensorRT library directory for linking
target_link_directories(${PROJECT_NAME}_lib PRIVATE ${TENSORRT_LIB_DIR})

add_executable(${PROJECT_NAME}_image inference_image.cpp)
add_executable(${PROJECT_NAME}_sequence inference_sequence.cpp)
add_executable(test_main test_main.cpp)

target_link_libraries(${PROJECT_NAME}_image ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_sequence ${PROJECT_NAME}_lib)
target_link_libraries(test_main ${PROJECT_NAME}_lib ${OpenCV_LIBRARIES} yaml-cpp)
