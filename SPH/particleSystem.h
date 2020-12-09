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
//#define DEBUG
//#define CPU_IMPL
#ifdef DEBUG
#define NUM_PARTICLES   1000
#else
#define NUM_PARTICLES   1000
#endif // DEBUG
#define CHUNK 4


#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "Eigen/Dense"
using namespace Eigen;

static const uint nextPow2(uint x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// Particle system class
class ParticleSystem {
public:
    ParticleSystem(uint numParticles, float3 boxDims, bool bUseOpenGL);
    ~ParticleSystem();

    enum ParticleConfig {
        CONFIG_RANDOM,
        CONFIG_GRID,
        _NUM_CONFIGS
    };

    enum ParticleArray {
        POSITION,
        VELOCITY,
    };

    void update(float deltaTime);
    void reset(ParticleConfig config);

    int    getNumParticles() const {
        return m_numParticles;
    }

    unsigned int getCurrentReadBuffer() const {
        return m_posVbo;
    }
    unsigned int getColorBuffer()       const {
        return m_colorVBO;
    }

    void* getCudaPosVBO()              const {
        return (void*)m_cudaPosVBO;
    }
    void* getCudaColorVBO()            const {
        return (void*)m_cudaColorVBO;
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

    void addSphere(int index, float* pos, float* vel, int r, float spacing);

protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);
    void updatePosVBO();

    void _initialize(int numParticles);
    void _finalize();

    void initGrid(uint* size, float spacing, float jitter, uint numParticles);
    void computeDensities();
    void computeForces();
    void particleCollisions();
    void integrate(float deltaTime);

    void zcomputeDensities();
    void zcomputeForces();
    void zparticleCollisions();
    uint coord2zIndex(Vector3i coord);
    Vector3i zIndex2coord(uint z_index);
    uint get_Z_index(Particle p);
    void constructGridArray();

    float guass_kernel(Vector3f rij, float h);
    Vector3f guass_kernel_gradient(Vector3f rij, float h);

    void computePressureIdeal(Particle& p);
    void computePressureTait(Particle& p);
    void computeDensity(Particle& pi, const Particle& pj);
    void computeForce(Particle& pi, const Particle& pj);
    void computeCollision(Particle& pi, const Particle& pj);

    std::vector<uint> getNeighbors(uint z_index);
protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;

    // CPU data
    std::vector<Particle> m_particles;      // Particle datastructure
    uint    m_z_grid_dim;       // dimension of the z-index grid
    uint  m_z_grid_size;        // size of the z-index grid
    Grid_item *m_z_grid;        // z-indexing grid array
    uint m_z_grid_prime_size;   // size of the compact z-index grid array
    Grid_item* m_z_grid_prime;  // compact z-indexing grid array
    float* m_hPos;              // particle positions

    // GPU data
    Particle* m_d_particles;    // GPU device particles datastructure
    Grid_item* m_d_B;           // GPU device full grid datastructure
    Grid_item* m_d_B_prime;     // GPU device compact grid datastructure
    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors

    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float* m_cudaColorVBO;      // these are the CUDA deviceMem Color

    // params
    SimParams m_params;
    float3 m_boxDims;

    StopWatchInterface* m_timer;

    uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
