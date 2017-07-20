cmake_minimum_required(VERSION 3.1)
project(matmul)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma -mavx -O3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx -O3")
endif()

add_executable(matmul "main.cpp")
set_target_properties(matmul PROPERTIES CXX_STANDARD 14)
