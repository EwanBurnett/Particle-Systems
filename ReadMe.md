# Particle Systems

A Project exploring how Parallelism can be used to simulate a large number of Particles, using C++, Compute Shaders and NVidia's [CUDA](https://developer.nvidia.com/cuda-toolkit). 

- This Project is licensed under the MIT License 

## Build Instructions

- Install the latest CUDA Toolkit for your platform
    - Windows 
        - Follow the download instructions on the [CUDA Toolkit Developer Portal](https://developer.nvidia.com/cuda-toolkit)
    

- Clone the Project by using 
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