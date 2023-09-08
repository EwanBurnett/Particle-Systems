cmake_minimum_required(VERSION 3.12)

project(Particles)

include(CheckLanguage)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
find_package(Threads REQUIRED)

set(BUILD_GAMES OFF)
set(BUILD_EXAMPLES OFF)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 3.0.0
)

FetchContent_MakeAvailable(raylib)

check_language(CUDA)
set(Particles_SOURCES "src/main.cpp")

# If CUDA is available, link the kernel
if(DEFINED CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    set(Particles_SOURCES ${Particles_SOURCES} "src/CUDAUpdate.cu")
	add_compile_definitions(CUDA_FOUND)
endif()


add_executable(${PROJECT_NAME} ${Particles_SOURCES})
target_link_libraries(${PROJECT_NAME} Threads::Threads raylib)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set_target_properties(${PROJECT_NAME} PROPERTIES
    CMAKE_CXX_STANDARD 20
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/"
	)
