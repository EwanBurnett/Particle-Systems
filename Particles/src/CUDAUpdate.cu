#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../include/CUDAUpdate.cuh"
#include "../include/Particle.h"
#include "../include/Vector3.h"

void* g_pDevPositions;
void* g_pDevVelocities;
void* g_pDevSpeeds;
void* g_pDevMasses;

#define UPDATE_BLOCK_SIZE 2048

__global__ void Update_Particles(Simulation::Vector3* pPositions, Simulation::Vector3* pVelocities, float* pSpeeds, float* pMasses, const size_t numParticles, const float deltaTime){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    size_t start = idx * UPDATE_BLOCK_SIZE;
    size_t end = start + UPDATE_BLOCK_SIZE;
    if(end > numParticles){
        return;
    }

    //Update the block
    for(size_t i = start; i < end; i++){
        const float k = 1.0f / pMasses[i];

        pPositions[i].x += (pVelocities[i].x * pSpeeds[i] * k * deltaTime);
        pPositions[i].y += (pVelocities[i].y * pSpeeds[i] * k * deltaTime);
        pPositions[i].z += (pVelocities[i].z * pSpeeds[i] * k * deltaTime);
    }
    //printf("\nblock: %d [ %d - %d ]\n", end, start, 100);

}

void CUDAInit(Particles* pParticles, const size_t count){
    //Allocate GPU memory for our Particles. 
    cudaMalloc(&g_pDevPositions, count * sizeof(Simulation::Vector3));
    cudaMalloc(&g_pDevVelocities, count * sizeof(Simulation::Vector3));
    cudaMalloc(&g_pDevSpeeds, count * sizeof(float));
    cudaMalloc(&g_pDevMasses, count * sizeof(float));
}

void CUDAUpdate(Particles* pParticles, const size_t count, const float deltaTime)
{
    //Launch the Particle Update kernel on the GPU 


    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = count / threadsPerBlock;

    cudaMemcpy(g_pDevPositions, pParticles->positions, count * sizeof(Simulation::Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevVelocities, pParticles->velocities, count * sizeof(Simulation::Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevSpeeds, pParticles->speeds, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevMasses, pParticles->masses, count * sizeof(float), cudaMemcpyHostToDevice);
    

    Update_Particles<<<blocksPerGrid, threadsPerBlock>>>((Simulation::Vector3*)g_pDevPositions, (Simulation::Vector3*)g_pDevVelocities, (float*)g_pDevSpeeds, (float*)g_pDevMasses, count, deltaTime);
    
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Error: %s\r", cudaGetErrorString(error));
        return;
    }
    //Copy the updated particle data back to the host 
    cudaMemcpy(pParticles->positions, g_pDevPositions, count * sizeof(Simulation::Vector3), cudaMemcpyDeviceToHost);


    
    printf("\r[CUDA] Updated %d particles in %fms", count, deltaTime);   
    
}
