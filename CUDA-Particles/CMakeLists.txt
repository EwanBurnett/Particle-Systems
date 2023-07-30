cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(CUDA-Particles LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} STATIC kernel.cu "inc/Particle.h" "inc/Vector3.h")

target_include_directories(${PROJECT_NAME} PUBLIC "inc/")

set_target_properties(${PROJECT_NAME} PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Lib"
	)
