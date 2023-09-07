#ifndef __VECTOR3_H
#define __VECTOR3_H

namespace Simulation
{
/**
 * @brief A simple 3-component vector, used to represent a point or direction in 3D Space. 
*/
struct Vector3 {
    Vector3(float X = 0, float Y = 0, float Z = 0) : x(X), y(Y), z(Z) {}
    float x, y, z;
};
}
#endif
