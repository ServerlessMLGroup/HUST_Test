project(json_test)
cmake_minimum_required(VERSION 3.2)

add_compile_options(-std=c++11)
include_directories(/usr/local/cuda-11.6/include)
find_package(CUDA REQUIRED)
INCLUDE(/opt/cmake-3.24.2/share/cmake-3.24/Modules/FindCUDA.cmake)
# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS -pthread)


aux_source_directory(. DIR_SRCS)

# 指定生成目标
add_executable(demo ${DIR_SRCS})

target_link_libraries(demo cuda)