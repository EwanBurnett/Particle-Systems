#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../include/CUDAUpdate.cuh"
#include "../include/Particle.h"

void* g_pDevPositions;
void* g_pDevVelocities;
void* g_pDevSpeeds;

__global__ void Update_Particles(){
    printf("\rUpdating with CUDA!");
}

void CUDAInit(Particles* pParticles, const size_t count){
    //Allocate GPU memory for our Particles. 
    cudaMalloc(&g_pDevPositions, count * sizeof(Vector3));
    cudaMemcpy(g_pDevPositions, pParticles->positions, count * sizeof(Vector3), cudaMemcpyHostToDevice);
}

void CUDAUpdate(Particles* pParticles, const size_t count, const float deltaTime)
{
    //Launch the Particle Update kernel on the GPU 
    Update_Particles<<<1, 1>>>();
    
    //Copy the updated particle data back to the host 
    
}
