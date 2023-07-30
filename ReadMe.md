# Particle Systems

A Project exploring how Parallelism can be used to simulate a large number of Particles, using C++, Compute Shaders and NVidia's [CUDA](https://developer.nvidia.com/cuda-toolkit). 

## Build Instructions
Clone the Project by using 
``` bash
git clone https://github.com/EwanBurnett/Particle-Systems/ 
cd Particle-Systems
```

### CMake
- Requires CMake 3.9 or greater ([CMake.org](https://cmake.org))

``` bash
mkdir build && cd build
cmake ..
```
- Build using your platform's toolchain
    - e.g. `make`

### Visual Studio
- Requires Visual Studio 2022 or greater 
    - Older versions of VS may work, but compatibility has not been tested.
- Set "Particles" as the Startup Project, and build. 