#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../include/CUDAUpdate.cuh"
#include "../include/Particle.h"
#include <chrono>

void* g_pDevPositions;
void* g_pDevVelocities;
void* g_pDevSpeeds;
void* g_pDevMasses;

#define UPDATE_BLOCK_SIZE 512

__global__ void Update_Particles(Vector3* pPositions, Vector3* pVelocities, float* pSpeeds, float* pMasses, const size_t numParticles, const float deltaTime){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    size_t start = idx * UPDATE_BLOCK_SIZE;
    size_t end = start + UPDATE_BLOCK_SIZE;
    if(end > numParticles){
        end = numParticles;
    }

    //Update the block
    for(size_t i = start; i < end; i++){
        const float k = 1.0f / pMasses[i];

        pPositions[i].x += (pVelocities[i].x * pSpeeds[i] * k * deltaTime);
        pPositions[i].y += (pVelocities[i].y * pSpeeds[i] * k * deltaTime);
        pPositions[i].z += (pVelocities[i].z * pSpeeds[i] * k * deltaTime);
    }

}

void CUDAInit(Particles* pParticles, const size_t count){
    //Allocate GPU memory for our Particles. 
    cudaMalloc(&g_pDevPositions, count * sizeof(Vector3));
    cudaMalloc(&g_pDevVelocities, count * sizeof(Vector3));
    cudaMalloc(&g_pDevSpeeds, count * sizeof(float));
    cudaMalloc(&g_pDevMasses, count * sizeof(float));
}

void CUDAUpdate(Particles* pParticles, const size_t count, const float deltaTime)
{
    //Launch the Particle Update kernel on the GPU 

    auto update_start = std::chrono::steady_clock::now();

    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(g_pDevPositions, pParticles->positions, count * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevVelocities, pParticles->velocities, count * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevSpeeds, pParticles->speeds, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pDevMasses, pParticles->masses, count * sizeof(float), cudaMemcpyHostToDevice);
    

    Update_Particles<<<threadsPerBlock, blocksPerGrid>>>((Vector3*)g_pDevPositions, (Vector3*)g_pDevVelocities, (float*)g_pDevSpeeds, (float*)g_pDevMasses, count, deltaTime);
    
    //Copy the updated particle data back to the host 
    cudaMemcpy(pParticles->positions, g_pDevPositions, count * sizeof(Vector3), cudaMemcpyDeviceToHost);

    const auto update_end = std::chrono::steady_clock::now();   //Finish the current frame
        float updateTime = std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_start).count() / 1000.0f; //Delta Time is in Milliseconds
    printf("\r[CUDA] Updated %d particles in %fms", count, updateTime);   
    
}
