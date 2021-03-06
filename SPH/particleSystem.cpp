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

 // OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "particleSystem.h"
#include "particles_kernel.cuh"
#include "particleSystem.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include <ctime>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, float3 boxDims, ParticleComputeMode mode) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hPos(0),
    m_boxDims(boxDims),
    m_compute_mode(mode),
    m_solverIterations(1) {
    // initialize grid
    m_h_B_dim = nextPow2((uint)(BOX_SIZE / (0.66666f * m_H)));
    m_h_B_size = m_h_B_dim * m_h_B_dim * m_h_B_dim;
    m_h_B = new Grid_item[m_h_B_size];

    // set simulation parameters
    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.gravity = make_float3(0.f, 0.f, 0.f);
    m_params.colliderRadius = 0.2f;
    m_params.boxMin.x = -boxDims.x / 2;
    m_params.boxMin.y = -boxDims.y / 2;
    m_params.boxMin.z = -boxDims.z / 2;
    m_params.boxMax.x = boxDims.x / 2;
    m_params.boxMax.y = boxDims.y / 2;
    m_params.boxMax.z = boxDims.z / 2;
    m_params.boxDims = boxDims;
    m_params.gridDim = m_h_B_dim;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem() {
    _finalize();
    m_numParticles = 0;
    delete[] m_h_B;
}

uint
ParticleSystem::createVBO(uint size) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// create a color ramp
void colorRamp(float t, float* r) {
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors - 1);
    int i = (int)t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void
ParticleSystem::_initialize(int numParticles) {
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles * 4];
    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));

    m_particles.resize(m_numParticles);

    // allocate GPU data
    allocateArray((void**)&m_d_params, sizeof(SimParams));
    allocateArray((void**)&m_d_particles, sizeof(Particle) * m_numParticles);
    allocateArray((void**)&m_d_B, sizeof(Grid_item) * m_h_B_size);
    allocateArray((void**)&m_d_B_prime, sizeof(Grid_item) * m_numParticles);

    unsigned int memSize = sizeof(float) * 4 * m_numParticles;
    m_posVbo = createVBO(memSize);
    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    m_colorVBO = createVBO(memSize);

    // fill color buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
    float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float* ptr = data;

    for (uint i = 0; i < m_numParticles; i++) {
        float t = i / (float)m_numParticles;
#if 0
        * ptr++ = rand() / (float)RAND_MAX;
        *ptr++ = rand() / (float)RAND_MAX;
        *ptr++ = rand() / (float)RAND_MAX;
#else
        colorRamp(t, ptr);
        ptr += 3;
#endif
        * ptr++ = 1.0f;
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);

    m_timer_start = std::chrono::steady_clock::now();
    m_global_time = 0.;

    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    std::string compute_mode = (m_compute_mode == SEQUENTIAL) ? "sequential" : (m_compute_mode == OMP_PARALLEL) ? "OMP" : "CUDA";
    std::string file_name = FILE_PREFIX +
        compute_mode + "_" +
        std::to_string(1900+now->tm_year) + "_" +
        std::to_string(now->tm_mon) + "_" + 
        std::to_string(now->tm_mday) + "_" + 
        std::to_string(now->tm_hour) + "_" + 
        std::to_string(now->tm_min) + ".txt";
    m_benchmark_file.open(file_name);
    m_benchmark_file << "SPH Particle Simulation Benchmark" << std::endl;
    m_benchmark_file << "Compute mode: " << compute_mode << std::endl;

    m_bInitialized = true;
}

void
ParticleSystem::_finalize() {
    assert(m_bInitialized);

    // free CPU data
    delete[] m_hPos;
    m_particles.clear();

    // free GPU data
    freeArray((void*)m_d_params);
    freeArray((void*)m_d_particles);
    freeArray((void*)m_d_B);
    freeArray((void*)m_d_B_prime);

    unregisterGLBufferObject(m_cuda_posvbo_resource);
    glDeleteBuffers(1, (const GLuint*)&m_posVbo);
    glDeleteBuffers(1, (const GLuint*)&m_colorVBO);

    m_benchmark_file.close();
}

//float ParticleSystem::guass_kernel(float3 rij, float h) {
//    float sigma = 1.f / (std::pow(PI_F, 1.5f) * h * h * h);
//    float q = norm(rij) / h;
//    if (q < 3.0f) {
//        return sigma * std::expf(-q * q);
//    }
//    else {
//        return 0.f;
//    }
//}
//
//float3 ParticleSystem::guass_kernel_gradient(float3 rij, float h) {
//    float3 grad = unit(rij);
//    float n = norm(rij);
//    float sigma = 1.f / (std::pow(PI_F, 1.5f) * h * h * h);
//    float q = n / h;
//    if (n > 1e-20f && q < 3.0f) {
//        float dq = -2.0f * q * sigma * std::expf(-q * q);
//        grad.x *= dq;
//        grad.y *= dq;
//        grad.z *= dq;
//        /*if (grad.x != grad.x || grad.y != grad.y || grad.z != grad.z) {
//            float3 u = unit(rij);
//            printf("grad <%f, %f, %f>, dq %f, q %f, sigma %f, unit <%f, %f, %f>\n", grad.x, grad.y, grad.z, dq, q, sigma, u.x, u.y, u.z);
//        }*/
//        return grad;
//    }
//    else {
//        return { 0.f,0.f,0.f };
//    }
//}

// https://matthias-research.github.io/pages/publications/sca03.pdf
// https://lucasschuermann.com/writing/implementing-sph-in-2d
void ParticleSystem::computePressureIdeal(Particle& p) {
    p.pressure = std::max(0.f, GAS_CONSTANT * (p.density - REST_DENS));
}

// https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
// https://en.wikipedia.org/wiki/Tait_equation
void ParticleSystem::computePressureTait(Particle& p) {
    const float cs = 88.5f;
    const float gamma = 7.f;
    const float B = (REST_DENS * cs * cs) / gamma;
    float quot = p.density / REST_DENS;
    float quot_exp_gamma = quot * quot * quot * quot * quot * quot * quot;
    p.pressure = B * ((quot_exp_gamma)-1.f);
}

void ParticleSystem::computeDensity(Particle& pi, const Particle& pj) {
    const float POLY6 = 315.f / (65.f * PI_F * std::pow(m_H, 9.f));
    Vector3f rij = pi.position - pj.position;
    float r2 = rij.squaredNorm();
    if (r2 < HSQ) {
        pi.density += pj.mass * POLY6 * std::pow(HSQ - r2, 3.f);
    }
}

void ParticleSystem::computeForce(Particle& pi, const Particle& pj) {
    const float SPIKY_GRAD = -45.f / (PI_F * std::pow(m_H, 6.f));
    const float VISC_LAP = 45.f / (PI_F * std::pow(m_H, 6.f));
    Vector3f rij = pi.position - pj.position;
    float r = rij.norm();
    if (r < m_H) {
        pi.force_press += -rij.normalized() * pj.mass * (pi.pressure + pj.pressure) / (2.f * pj.density) * SPIKY_GRAD * std::pow(m_H - r, 2.f);
        pi.force_visc += VISC * pj.mass * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (m_H - r);
    }
}

void ParticleSystem::computeCollision(Particle& pi, const Particle& pj) {
    Vector3f vij, rij;
    float dij;
    vij = pi.velocity - pj.velocity;
    rij = pi.position - pj.position;
    dij = rij.norm();
    if (dij <= COLLISION_PARAM * 2 * pi.radius && rij.dot(vij) < 0) {
        pi.delta_velocity += (pj.mass * (1.f + RESTITUTION)) * (rij.dot(vij) / (dij * dij)) * rij;
        pi.collision_count++;
    }
}

std::vector<uint> ParticleSystem::getNeighbors(uint z_index) {
    std::vector<uint> neighbors;
    Vector3i coords = zIndex2coord(z_index);
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int x = coords.x() + dx;
                int y = coords.y() + dy;
                int z = coords.z() + dz;
                if (0 <= x && x < m_h_B_dim &&
                    0 <= y && y < m_h_B_dim &&
                    0 <= z && z < m_h_B_dim) {
                    neighbors.push_back(coord2zIndex(Vector3i({ x,y,z })));
                }
            }
        }
    }
    return neighbors;
}

void ParticleSystem::computeDensities() {
    for (Particle& pi : m_particles) {
        pi.density = 0.f;
        for (Particle& pj : m_particles) {
            computeDensity(pi, pj);
        }
        computePressureIdeal(pi);
    }
}

void ParticleSystem::zcomputeDensities() {
    // loop through each grid block, and for each only compute using its particles
#pragma omp parallel for schedule(static, OMP_CHUNK)
    for (int block = 0; block < m_h_B_size; block++) {
        if (m_h_B[block].nParticles == 0) continue;
        for (int i = m_h_B[block].start; i < m_h_B[block].start + m_h_B[block].nParticles; i++) {
            Particle& pi = m_particles[i];
            std::vector<uint> neighbors = getNeighbors(block);
            pi.density = 0.f;
            for (uint neighbor : neighbors) {
                if (m_h_B[neighbor].nParticles == 0) continue;
                for (int j = m_h_B[neighbor].start; j < m_h_B[neighbor].start + m_h_B[neighbor].nParticles; j++) {
                    Particle& pj = m_particles[j];
                    computeDensity(pi, pj);
                }
            }
            computePressureIdeal(pi);
        }
    }
}

void ParticleSystem::computeForces() {
    for (Particle& pi : m_particles) {
        pi.force_press = { 0.f, 0.f, 0.f };
        pi.force_visc = { 0.f, 0.f, 0.f };
        for (Particle& pj : m_particles) {
            computeForce(pi, pj);
        }
    }
}

void ParticleSystem::zcomputeForces() {
#pragma omp parallel for schedule(static, OMP_CHUNK)
    for (int block = 0; block < m_h_B_size; block++) {
        if (m_h_B[block].nParticles == 0) continue;
        for (int i = m_h_B[block].start; i < m_h_B[block].start + m_h_B[block].nParticles; i++) {
            Particle& pi = m_particles[i];
            std::vector<uint> neighbors = getNeighbors(block);
            pi.force_press = { 0.f, 0.f, 0.f };
            pi.force_visc = { 0.f, 0.f, 0.f };
            for (uint neighbor : neighbors) {
                if (m_h_B[neighbor].nParticles == 0) continue;
                for (int j = m_h_B[neighbor].start; j < m_h_B[neighbor].start + m_h_B[neighbor].nParticles; j++) {
                    Particle& pj = m_particles[j];
                    computeForce(pi, pj);
                }
            }
        }
    }
}

void ParticleSystem::particleCollisions() {
    // detect collisions
    for(int i = 0; i < m_particles.size(); i++) {
        Particle& pi = m_particles[i];
        pi.delta_velocity = { 0.f, 0.f, 0.f };
        pi.collision_count = 0;
        for (Particle& pj : m_particles) {
            if (pi.index == pj.index) continue;
            computeCollision(pi, pj);
        }
        pi.delta_velocity = -pi.delta_velocity / (pi.mass * (1 + pi.collision_count));
    }
}

void ParticleSystem::zparticleCollisions() {
    // detect collisions
#pragma omp parallel for schedule(static, OMP_CHUNK)
    for (int block = 0; block < m_h_B_size; block++) {
        if (m_h_B[block].nParticles == 0) continue;
        for (int i = m_h_B[block].start; i < m_h_B[block].start + m_h_B[block].nParticles; i++) {
            Particle& pi = m_particles[i];
            std::vector<uint> neighbors = getNeighbors(block);
            pi.delta_velocity = { 0.f, 0.f, 0.f };
            pi.collision_count = 0;
            for (uint neighbor : neighbors) {
                if (m_h_B[neighbor].nParticles == 0) continue;
                for (int j = m_h_B[neighbor].start; j < m_h_B[neighbor].start + m_h_B[neighbor].nParticles; j++) {
                    Particle& pj = m_particles[j];
                    if (pi.index == pj.index) continue;
                    computeCollision(pi, pj);
                }
            }
            pi.delta_velocity = -pi.delta_velocity / (pi.mass * (1 + pi.collision_count));
        }
    }
}

void ParticleSystem::integrate(float deltaTime) {
    for (int i = 0; i < m_particles.size(); i++) {
        Particle& p = m_particles[i];
        Vector3f force_grav = { 0.f, GRAVITY * G_MODIFIER * p.density, 0.f };
        Vector3f force = p.force_press + p.force_visc + force_grav;
        Vector3f accel = force / p.density;
        p.velocity += deltaTime * accel + p.delta_velocity; 
        p.position += deltaTime * p.velocity;

        // bounds check in X
        if (p.position.x() - EPS_F < m_params.boxMin.x) {
            p.position.x() = m_params.boxMin.x + EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }
        if (p.position.x() + EPS_F > m_params.boxMax.x) {
            p.position.x() = m_params.boxMax.x - EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }

        // bounds check in Y
        if (p.position.y() - EPS_F < m_params.boxMin.y) {
            p.position.y() = m_params.boxMin.y + EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }
        if (p.position.y() + EPS_F > m_params.boxMax.y) {
            p.position.y() = m_params.boxMax.y - EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }

        // bounds check in Z
        if (p.position.z() - EPS_F < m_params.boxMin.z) {
            p.position.z() = m_params.boxMin.z + EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }
        if (p.position.z() + EPS_F > m_params.boxMax.z) {
            p.position.z() = m_params.boxMax.z - EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }

        m_hPos[p.index * 4 + 0] = p.position.x();
        m_hPos[p.index * 4 + 1] = p.position.y();
        m_hPos[p.index * 4 + 2] = p.position.z();
        m_hPos[p.index * 4 + 3] = 1.0f;
    }
}

void ParticleSystem::zintegrate(float deltaTime) {
#pragma omp parallel for schedule(static, OMP_INTEGRATE_CHUNK)
    for (int i = 0; i < m_particles.size(); i++) {
        Particle& p = m_particles[i];
        Vector3f force_grav = { 0.f, GRAVITY * G_MODIFIER * p.density, 0.f };
        Vector3f force = p.force_press + p.force_visc + force_grav;
        Vector3f accel = force / p.density;
        p.velocity += deltaTime * accel + p.delta_velocity;
        p.position += deltaTime * p.velocity;

        // bounds check in X
        if (p.position.x() - EPS_F < m_params.boxMin.x) {
            p.position.x() = m_params.boxMin.x + EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }
        if (p.position.x() + EPS_F > m_params.boxMax.x) {
            p.position.x() = m_params.boxMax.x - EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }

        // bounds check in Y
        if (p.position.y() - EPS_F < m_params.boxMin.y) {
            p.position.y() = m_params.boxMin.y + EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }
        if (p.position.y() + EPS_F > m_params.boxMax.y) {
            p.position.y() = m_params.boxMax.y - EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }

        // bounds check in Z
        if (p.position.z() - EPS_F < m_params.boxMin.z) {
            p.position.z() = m_params.boxMin.z + EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }
        if (p.position.z() + EPS_F > m_params.boxMax.z) {
            p.position.z() = m_params.boxMax.z - EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }

        m_hPos[p.index * 4 + 0] = p.position.x();
        m_hPos[p.index * 4 + 1] = p.position.y();
        m_hPos[p.index * 4 + 2] = p.position.z();
        m_hPos[p.index * 4 + 3] = 1.0f;
    }
}

// coord components must be 10 bits 
uint ParticleSystem::coord2zIndex(Vector3i coord) {
    uint x_grid = coord.x();
    uint y_grid = coord.y();
    uint z_grid = coord.z();

    // interleave the grid indices to find the final z-index
   // x bits
    x_grid = (x_grid | (x_grid << 16)) & 0x030000FF;
    x_grid = (x_grid | (x_grid << 8)) & 0x0300F00F;
    x_grid = (x_grid | (x_grid << 4)) & 0x030C30C3;
    x_grid = (x_grid | (x_grid << 2)) & 0x09249249;
    // y bits
    y_grid = (y_grid | (y_grid << 16)) & 0x030000FF;
    y_grid = (y_grid | (y_grid << 8)) & 0x0300F00F;
    y_grid = (y_grid | (y_grid << 4)) & 0x030C30C3;
    y_grid = (y_grid | (y_grid << 2)) & 0x09249249;
    // z bits
    z_grid = (z_grid | (z_grid << 16)) & 0x030000FF;
    z_grid = (z_grid | (z_grid << 8)) & 0x0300F00F;
    z_grid = (z_grid | (z_grid << 4)) & 0x030C30C3;
    z_grid = (z_grid | (z_grid << 2)) & 0x09249249;

    return x_grid | (y_grid << 1) | (z_grid << 2);
}

// 30  bit number -> 10 bit number
uint collapseEvery3(uint x) {
    uint res = 0;
    for (int i = 0; i < 10; i++) {
        uint b = ((x >> i * 3) & 1);
        res = res | (b << i); 
    }
    return res;
}

Vector3i ParticleSystem::zIndex2coord(uint z_index) {
    uint x = collapseEvery3(z_index);
    uint y = collapseEvery3(z_index >> 1);
    uint z = collapseEvery3(z_index >> 2);
    return Vector3i(x, y, z);
}

uint ParticleSystem::get_Z_index(Particle p) {
    // find the section of the grid the particle is in
    float posX = p.position.x() - m_params.boxMin.x;
    float posY = p.position.y() - m_params.boxMin.y;
    float posZ = p.position.z() - m_params.boxMin.z;
    Vector3i coord;
    coord.x() = floor((posX / m_boxDims.x) * m_h_B_dim);
    coord.y() = floor((posY / m_boxDims.y) * m_h_B_dim);
    coord.z() = floor((posZ / m_boxDims.z) * m_h_B_dim);
    return coord2zIndex(coord);
}


bool cmpParticles(Particle pi, Particle pj) {
    return (pi.zindex < pj.zindex);
}

void ParticleSystem::zMapZindex() {
    // set each particle's z-index
    for (Particle& p : m_particles) {
        p.zindex = get_Z_index(p);
    }
}

void ParticleSystem::zSortParticles() {
    // sort the particles vector according to the z index
    std::sort(m_particles.begin(), m_particles.end(), cmpParticles);
}

void ParticleSystem::zConstructBGrid() {
    // clear prev grid array
    std::memset(m_h_B, 0, m_h_B_size * sizeof(Grid_item));
    // set the grid array where each item has the starting index into the particles
    // vector and the number of particles that block in the grid contains
    long long grid_dex = -1;
    for (int i = 0; i < m_particles.size(); i++) {
        Particle p = m_particles[i];
        //printf("p is at zindex %d\n", p.zindex);
        unsigned long long zind = p.zindex;
        if (zind != grid_dex) {
            // found a new block of particles
            grid_dex = zind;
            m_h_B[grid_dex].start = i;
            m_h_B[grid_dex].nParticles = 1;
            //printf("Found a new block at %d\n", i);  
        }
        else {
            m_h_B[grid_dex].nParticles++; // still in same block, increment particles
            //printf("Incremented %d to %d particles\n", grid_dex, m_h_B[grid_dex].nParticles);
        }
    }
}

void ParticleSystem::zConstructGridArray() {
    // compact z_grid
    std::vector<Grid_item> grid;
    for (int i = 0; i < m_h_B_size; i++) {
        uint iter = 0;
        while (iter < m_h_B[i].nParticles) {
            Grid_item gi;
            gi.start = iter + m_h_B[i].start;
            gi.nParticles = std::min(GRID_COMPACT_WIDTH, m_h_B[i].nParticles - iter);
            grid.push_back(gi);
            iter += GRID_COMPACT_WIDTH;
        }
    }
    m_h_B_prime_size = grid.size();
    m_h_B_prime = new Grid_item[m_h_B_prime_size];
    std::memcpy((void*)m_h_B_prime, grid.data(), m_h_B_prime_size * sizeof(Grid_item));
}

void ParticleSystem::constructGridArray() {
    // set each particle's z-index
    for (Particle& p : m_particles) {
        p.zindex = get_Z_index(p);
    }
    // sort the particles vector according to the z index
    std::sort(m_particles.begin(), m_particles.end(), cmpParticles);
    // clear prev grid array
    std::memset(m_h_B, 0, m_h_B_size * sizeof(Grid_item));
    // set the grid array where each item has the starting index into the particles
    // vector and the number of particles that block in the grid contains
    long long grid_dex = -1;
    for (int i = 0; i < m_particles.size(); i++) {
        Particle p = m_particles[i];
        //printf("p is at zindex %d\n", p.zindex);
        unsigned long long zind = p.zindex;
        if (zind != grid_dex) {
            // found a new block of particles
            grid_dex = zind;
            m_h_B[grid_dex].start = i;
            m_h_B[grid_dex].nParticles = 1;
            //printf("Found a new block at %d\n", i);  
        }
        else {
            m_h_B[grid_dex].nParticles++; // still in same block, increment particles
            //printf("Incremented %d to %d particles\n", grid_dex, m_h_B[grid_dex].nParticles);
        }
    }
    copyArrayToDevice((void*)m_d_B, m_h_B, m_h_B_size * sizeof(Grid_item));

    // compact z_grid
    std::vector<Grid_item> grid;
    for (int i = 0; i < m_h_B_size; i++) {
        uint iter = 0;
        while (iter < m_h_B[i].nParticles) {
            Grid_item gi;
            gi.start = iter + m_h_B[i].start;
            gi.nParticles = std::min(GRID_COMPACT_WIDTH, m_h_B[i].nParticles - iter);
            grid.push_back(gi);
            iter += GRID_COMPACT_WIDTH;
        }
    }
    m_h_B_prime_size = grid.size();
    m_h_B_prime = new Grid_item[m_h_B_prime_size];
    std::memcpy((void*)m_h_B_prime, grid.data(), m_h_B_prime_size * sizeof(Grid_item));

    // allocate cuda B_prime array
    copyArrayToDevice((void*)m_d_B_prime, m_h_B_prime, m_h_B_prime_size * sizeof(Grid_item));
}

void ParticleSystem::constructGridArrayAlt() {
    // compact z_grid
    std::vector<Grid_item> grid;
    for (int i = 0; i < m_h_B_size; i++) {
        uint iter = 0;
        while (iter < m_h_B[i].nParticles) {
            Grid_item gi;
            gi.start = iter + m_h_B[i].start;
            gi.nParticles = std::min(GRID_COMPACT_WIDTH, m_h_B[i].nParticles - iter);
            grid.push_back(gi);
            iter += GRID_COMPACT_WIDTH;
        }
    }
    m_h_B_prime_size = grid.size();
    m_h_B_prime = new Grid_item[m_h_B_prime_size];
    std::memcpy((void*)m_h_B_prime, grid.data(), m_h_B_prime_size * sizeof(Grid_item));

    // allocate cuda B_prime array
    copyArrayToDevice((void*)m_d_B_prime, m_h_B_prime, m_h_B_prime_size * sizeof(Grid_item));
}

void printZGrid(Grid_item *m_h_B, Grid_item *m_h_B_prime, uint m_h_B_prime_size) {
    printf("\ngrid item array \n");
    for (int i = 0; i < 10; i++) {
        printf("i:%d,start:%u,size:%u;    ", i, m_h_B[i].start, m_h_B[i].nParticles);
    }
    printf("\ngrid item prime array\n");
    for (int i = 0; i < m_h_B_prime_size; i++) {
        if(m_h_B_prime[i].nParticles != 0)
        printf("i:%d,start:%u,size:%u;    ", i, m_h_B_prime[i].start, m_h_B_prime[i].nParticles);
    }
}

void ParticleSystem::dumpBenchmark(long long fps, long long d_t, long long f_t, long long pc_t, long long i_t, long long t_t) {
    m_timer_curr = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_timer_curr - m_timer_start);
    if (duration.count() > BENCHMARK_FREQ) {
        m_timer_start = m_timer_curr;
        m_global_time += duration.count();
        m_benchmark_file << m_global_time / 1000 << "sec" << "\ttotal:" << t_t << 
            "ns,\t\tdens:" << d_t << 
            "ns,\t\tforce:" << f_t << 
            "ns,\t\tcollision:" << pc_t << 
            "ns,\t\tintegrate:" << i_t << 
            "ns,\t\tframes:" << fps <<
            "frames" << std::endl;
    }
}

void ParticleSystem::dumpBenchmark(long long fps, long long z_t, long long s_t, long long cbg_t, long long cbpg_t, long long d_t, long long f_t, long long pc_t, long long i_t, long long t_t, long long cp_t) {
    m_timer_curr = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_timer_curr - m_timer_start);
    if (duration.count() > BENCHMARK_FREQ) {
        m_timer_start = m_timer_curr;
        m_global_time += duration.count();
        m_benchmark_file << m_global_time / 1000 << "sec" << "\ttotal:" << t_t << 
            "ns,\t\tcopying:" << cp_t <<
            "ns,\t\tz-index:" << z_t <<
            "ns,\t\tsort:" << s_t <<
            "ns,\t\tb-grid:"<< cbg_t <<
            "ns,\t\tb'-grid:" << cbpg_t <<
            "ns,\t\tdens:" << d_t << 
            "ns,\t\tforce:" << f_t << 
            "ns,\t\tcollision:" << pc_t << 
            "ns,\t\tintegrate:" << i_t << 
            "ns,\t\tframes:" << fps <<
            "frames" << std::endl;
    }
}

// step the simulation
void
ParticleSystem::update(float deltaTime, float fps) {
    assert(m_bInitialized);
    omp_set_num_threads(omp_get_max_threads());
    copyArrayToDevice((void*)m_d_params, &m_params, sizeof(SimParams));
    //copyArrayToDevice((void*)m_d_B, m_h_B, m_h_B_size * sizeof(Grid_item));
    for (int iter = 0; iter < m_solverIterations; iter++) {
        long long zi_time, s_time, cbg_time, cbpg_time, d_time, f_time, pc_time, i_time, total_time;
        long long cp_time = 0;
        zi_time = s_time = cbg_time = cbpg_time = d_time = f_time = pc_time = i_time = total_time = 0;
        auto s = std::chrono::steady_clock::now();
        if (m_compute_mode == SEQUENTIAL) {
            // SEQUENTIAL IMPLEMENTATION
            TIME_FUNCTION(d_time, computeDensities());

            TIME_FUNCTION(f_time, computeForces());

            TIME_FUNCTION(pc_time, particleCollisions());

            TIME_FUNCTION(i_time, integrate(deltaTime));
        }
        else if (m_compute_mode == OMP_PARALLEL) {
            // OPENMP IMPLEMENTATION
            // Map each particle to their z index
            TIME_FUNCTION(zi_time, zMapZindex());

            // Sort the particles by zindex
            TIME_FUNCTION(s_time, zSortParticles());

            // construct the B grid array
            TIME_FUNCTION(cbg_time, zConstructBGrid());

            // construct the B' grid array
            TIME_FUNCTION(cbpg_time, zConstructGridArray());

            // N^2 algorithm for calculating density for each particle, computes pressure as well
            TIME_FUNCTION(d_time, zcomputeDensities());

            // computes pressure and gravity force contribution on each particle
            TIME_FUNCTION(f_time, zcomputeForces());

            // find particle collisions
            TIME_FUNCTION(pc_time, zparticleCollisions());

            // integrates velocity and position based on forces
            TIME_FUNCTION(i_time, zintegrate(deltaTime));

            // free z_grid_prime (b_prime)
            delete[] m_h_B_prime;
        }
        else {
            assert(m_compute_mode == CUDA_PARALLEL);
            // CUDA IMLPEMENTATION
            // Map each particle to their z index
            TIME_FUNCTION(zi_time, cudaMapZIndex(m_d_particles, m_numParticles, m_d_params));

            // Sort the particles by zindex
            TIME_FUNCTION(s_time, cudaSortParticles(m_d_particles, m_numParticles));

            // construct the B grid array
            TIME_FUNCTION(cbg_time, cudaConstructBGrid(m_d_particles, m_numParticles, m_d_B, m_h_B_size, m_d_params));

            // construct the B' grid array
            TIME_FUNCTION(cbpg_time, cudaConstructGridArray(m_d_particles, m_numParticles, m_d_B, m_h_B_size, &m_d_B_prime, &m_h_B_prime_size, m_d_params));

            // copmute density and pressure for every particle
            TIME_FUNCTION(d_time, cudaComputeDensities(m_d_particles, m_numParticles, m_d_B, m_h_B_size, m_d_B_prime, m_h_B_prime_size, m_d_params));

            // computes pressure and viscosity force contribution on each particle
            TIME_FUNCTION(f_time, cudaComputeForces(m_d_particles, m_numParticles, m_d_B, m_h_B_size, m_d_B_prime, m_h_B_prime_size, m_d_params));

            // find particle collisions
            TIME_FUNCTION(pc_time, cudaParticleCollisions(m_d_particles, m_numParticles, m_d_B, m_h_B_size, m_d_B_prime, m_h_B_prime_size, m_d_params));

            // integrates velocity and position based on forces
            float* m_dPos = (float*)mapGLBufferObject(&m_cuda_posvbo_resource);
            TIME_FUNCTION(i_time, cudaIntegrate(m_dPos, deltaTime, m_d_particles, m_numParticles, m_d_params));

            auto __start = std::chrono::steady_clock::now();
            unmapGLBufferObject(m_cuda_posvbo_resource);
            auto __end = std::chrono::steady_clock::now();
            cp_time = (__end - __start).count();
        }
        auto e = std::chrono::steady_clock::now();
        total_time = (e - s).count();
        // dump to file
        if (m_compute_mode == CUDA_PARALLEL || m_compute_mode == OMP_PARALLEL) {
            dumpBenchmark(fps, zi_time, s_time, cbg_time, cbpg_time, d_time, f_time, pc_time, i_time, total_time, cp_time);
        }
        else {
            dumpBenchmark(fps, d_time, f_time, pc_time, i_time, total_time);
        }
    }

    if (m_compute_mode == SEQUENTIAL || m_compute_mode == OMP_PARALLEL) {
        // update the vertex buffer object
        updatePosVBO();
    }
}

void
ParticleSystem::dumpParticles(uint start, uint count) {
    // debug

    for (uint i = start; i < start + count; i++) {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
    }
}

void ParticleSystem::updatePosVBO() {
    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * 4 * sizeof(float), m_hPos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

inline float frand() {
    return rand() / (float)RAND_MAX;
}

void
ParticleSystem::initGrid(uint* size, float spacing, float jitter, uint numParticles) {
    srand(1973);

    for (uint z = 0; z < size[2]; z++) {
        for (uint y = 0; y < size[1]; y++) {
            for (uint x = 0; x < size[0]; x++) {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles) {
                    float w, h, d;
                    w = m_boxDims.x;
                    h = m_boxDims.y;
                    d = m_boxDims.z;
                    Particle& p = m_particles.at(i);
                    p.index = i;
                    p.position = { (spacing * x) + m_params.particleRadius + m_params.boxMin.x + (w * frand() - w / 2) * jitter ,
                                   (spacing * y) + m_params.particleRadius + m_params.boxMin.y + (h * frand() - h / 2) * jitter ,
                                   (spacing * z) + m_params.particleRadius + m_params.boxMin.z + (d * frand() - d / 2) * jitter };
                    p.velocity = { 0.f,0.f,0.f };
                    p.force_press = { 0.f,0.f,0.f };
                    p.force_visc = { 0.f,0.f,0.f };
                    p.mass = MASS;
                    p.density = 0.f;
                    p.pressure = 0.f;
                    p.radius = m_params.particleRadius;
                    m_hPos[p.index * 4 + 0] = p.position.x();
                    m_hPos[p.index * 4 + 1] = p.position.y();
                    m_hPos[p.index * 4 + 2] = p.position.z();
                    m_hPos[p.index * 4 + 3] = 1.0f;
                }
            }
        }
    }
    updatePosVBO();
}

void
ParticleSystem::reset(ParticleConfig config) {
    switch (config) {
    default:
    case CONFIG_RANDOM:
    {
        int p = 0, v = 0;

        for (uint i = 0; i < m_numParticles; i++) {
            float w, h, d;
            w = m_boxDims.x;
            h = m_boxDims.y;
            d = m_boxDims.z;
            Particle& p = m_particles.at(i);
            p.index = i;
            p.position = { w * frand() - w / 2,
                           h * frand() - h / 2,
                           d * frand() - d / 2 };
            p.velocity = { 0.f,0.f,0.f };
            p.force_press = { 0.f,0.f,0.f };
            p.force_visc = { 0.f,0.f,0.f };
            p.mass = MASS;
            p.density = 0.f;
            p.pressure = 0.f;
            p.radius = m_params.particleRadius;
            m_hPos[p.index * 4 + 0] = p.position.x();
            m_hPos[p.index * 4 + 1] = p.position.y();
            m_hPos[p.index * 4 + 2] = p.position.z();
            m_hPos[p.index * 4 + 3] = 1.f;
        }
    }
    break;

    case CONFIG_GRID:
    {
        float jitter = m_params.particleRadius * 0.01f;
        uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint gridSize[3];
        gridSize[0] = gridSize[1] = gridSize[2] = s;
        initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numParticles);
    }
    break;
    }
    updatePosVBO();
    copyArrayToDevice((void*)m_d_particles, m_particles.data(), m_numParticles * sizeof(Particle));
}

void
ParticleSystem::addSphere(int start, float* pos, float* vel, int r, float spacing) {
    uint index = start;
    float w, h, d;
    w = m_boxDims.x;
    h = m_boxDims.y;
    d = m_boxDims.z;
    std::sort(m_particles.begin(), m_particles.end(),
        [](const Particle e0, const Particle e1) {
            return e0.index < e1.index;
        });

    for (int z = -r; z <= r; z++) {
        for (int y = -r; y <= r; y++) {
            for (int x = -r; x <= r; x++) {
                float dx = x * spacing;
                float dy = y * spacing;
                float dz = z * spacing;
                float l = sqrtf(dx * dx + dy * dy + dz * dz);
                float jitter = m_params.particleRadius * 0.01f;

                if ((l <= m_params.particleRadius * 2.0f * r) && (index < m_numParticles)) {
                    Particle& p = m_particles.at(index);
                    p.position = { pos[0] + dx + (w * frand() - w / 2) * jitter,
                                   pos[1] + dy + (h * frand() - h / 2) * jitter,
                                   pos[2] + dz + (d * frand() - d / 2) * jitter};
                    m_hPos[p.index * 4 + 0] = p.position.x();
                    m_hPos[p.index * 4 + 1] = p.position.y();
                    m_hPos[p.index * 4 + 2] = p.position.z();
                    m_hPos[p.index * 4 + 3] = 1.f;
                    index++;

                }
            }
        }
    }
    updatePosVBO();
    copyArrayToDevice((void*)m_d_particles, m_particles.data(), m_numParticles * sizeof(Particle));
}
