/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
#include "Eigen/Core"
using namespace Eigen;
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float particleRadius;

    float3 boxMin;
    float3 boxMax;
};

struct Particle {
    uint index;
    Vector3f position;
    Vector3f velocity;
    Vector3f delta_velocity;
    Vector3f force;
    float mass;
    float density;
    float pressure;
    float radius;
    int collision_count;
    uint zindex;
};

#endif
