#ifndef __CUDA_UPDATE_CUH
#define __CUDA_UPDATE_CUH
#include <Particle.h>

void CUDAInit(Particles* pParticles, const size_t count);
void CUDAUpdate(Particles* pParticles, const size_t count, const float deltaTime);

#endif
