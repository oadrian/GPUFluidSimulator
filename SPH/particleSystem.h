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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0
#define DEFAULT_MASS (1e-2f)
#define REST_DENSITY 997.f
#define DEFAULT_PRESSURE 0.f
#define GRAVITY -9.81f
#define SMOOTH_WIDTH 0.25f
#define Kp 2000.f
#define PI_F         3.141592654f
#define EPS_F        0.00001f

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

inline float norm(float3 v) { return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
inline float3 unit(float3 v) {
    const float n = norm(v);
    return { v.x / n, v.y / n, v.z / n };
}

class Particle {
public:
    Particle() {}
    Particle(float4 pos, float3 vel, float3 f, float rad) :
        position(pos),
        velocity(vel),
        force(f),
        mass(DEFAULT_MASS),
        density(REST_DENSITY),
        pressure(DEFAULT_PRESSURE),
        radius(rad) {}
    ~Particle() {}
    uint index;
    float4 position;
    float3 velocity;
    float3 force;
    float mass;
    float density;
    float pressure;
    float radius;
};

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, float3 boxDims, bool bUseOpenGL);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };

        void update(float deltaTime);
        void reset(ParticleConfig config);

        int    getNumParticles() const
        {
            return m_numParticles;
        }

        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }

        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }
        void *getCudaColorVBO()            const
        {
            return (void *)m_cudaColorVBO;
        }

        void dumpParticles(uint start, uint count);

        void setIterations(int i) {
            m_solverIterations = i;
        }
        void setGravity(float x) {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }
        void setColliderPos(float3 x) {
            m_params.colliderPos = x;
        }

        float3 getColliderPos() {
            return m_params.colliderPos;
        }
        float getColliderRadius() {
            return m_params.colliderRadius;
        }
        float getParticleRadius() {
            return m_params.particleRadius;
        }
        float3 getBoxMin() {
            return m_params.boxMin;
        }
        float3 getBoxMax() {
            return m_params.boxMax;
        }

        void addSphere(int index, float *pos, float *vel, int r, float spacing);

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);
        void updatePosVBO();

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);
        void computeDensities();
        void computeForces();
        void integrate(float deltaTime);

        float guass_kernel(float3 rij, float h);
        float3 guass_kernel_gradient(float3 rij, float h);

        float pressure_ideal_gas(const Particle &p);
        float pressure_tait_eq(const Particle &p);

    protected: // data
        bool m_bInitialized, m_bUseOpenGL;
        uint m_numParticles;

        // CPU data
        std::vector<Particle> m_particles;      // Particle datastructure
        float *m_hPos;              // particle positions

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint   m_colorVBO;          // vertex buffer object for colors

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

        // params
        SimParams m_params;
        float3 m_boxDims;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
