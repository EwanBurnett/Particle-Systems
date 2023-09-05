cmake_minimum_required(VERSION 3.12)

project(Particles)

include(CheckLanguage)
find_package(Threads REQUIRED)
check_language(CUDA)

set(Particles_SOURCES "src/main.cpp")

# If CUDA is available, link the kernel
if(DEFINED CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    set(Particles_SOURCES ${Particles_SOURCES} "src/CUDAUpdate.cu")
	add_compile_definitions(CUDA_FOUND)
endif()



add_executable(${PROJECT_NAME} ${Particles_SOURCES})
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set_target_properties(${PROJECT_NAME} PROPERTIES
    CMAKE_CXX_STANDARD 20
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/"
	)
