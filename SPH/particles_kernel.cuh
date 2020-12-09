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

#define REST_DENS 1000.f
#define GAS_CONSTANT 2000.f
#define m_H 0.1f
#define HSQ m_H * m_H
#define MASS 65.f
#define VISC 250.f
#define GRAVITY -9.81f
#define G_MODIFIER 11000
#define PI_F         3.141592654f
#define EPS_F        0.00001f
#define RESTITUTION 0.f
#define COLLISION_PARAM 1.0
#define BOX_SIZE 1.f
#define GRID_COMPACT_WIDTH 32u

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
    Vector3f force_press;
    Vector3f force_visc;
    float mass;
    float density;
    float pressure;
    float radius;
    int collision_count;
    uint zindex;
};

struct Grid_item {
    uint nParticles;
    uint start;
};

#endif
