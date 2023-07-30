cmake_minimum_required(VERSION 3.12)

project(Particles)

add_executable(${PROJECT_NAME} main.cpp)

find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA-Particles Threads::Threads)

set_target_properties(${PROJECT_NAME} PROPERTIES
	CXX_STANDARD 20
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/"
	)
