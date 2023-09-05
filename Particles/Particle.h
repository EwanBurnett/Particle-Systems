#ifndef __PARTICLE_H
#define __PARTICLE_H

#include "Vector3.h"

/**
 * @brief Particles are represented in 3D space, and have a Position, and a Velocity. 
 * @remark Particles are stored in a Structure of Arrays, to increase cache locality. 
 */
struct Particles {
    Particles(size_t num_particles) {
        //Allocate contiguous arrays for each element 
        positions = new Vector3[num_particles];
        velocities = new Vector3[num_particles];
        masses = new float[num_particles];
        speeds = new float[num_particles];
    }
    ~Particles() {
        //Ensure all allocated memory is freed upon destruction. 
        delete[] positions;
        delete[] velocities;
        delete[] masses;
        delete[] speeds;
    }

    Vector3* positions;
    Vector3* velocities;
    float* masses;
    float* speeds;
};


#endif