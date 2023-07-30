#ifndef __PARTICLE_H
#define __PARTICLE_H

#include "Vector3.h"

/**
 * @brief Particles are represented in 3D space, and have a Position, and a Velocity. 
 */
struct Particle {
    Particle() : mass(1.0f), speed(1.0f) {}

    Vector3 position;
    Vector3 velocity;
    float mass;
    float speed;
};

#endif