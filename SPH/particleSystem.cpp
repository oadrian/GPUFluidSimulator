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

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, float3 boxDims, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_boxDims(boxDims),
    m_timer(NULL),
    m_solverIterations(1)
{
    // set simulation parameters
    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;
    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.boxMin.x = - boxDims.x / 2;
    m_params.boxMin.y = - boxDims.y / 2;
    m_params.boxMin.z = - boxDims.z / 2;
    m_params.boxMax.x = boxDims.x / 2;
    m_params.boxMax.y = boxDims.y / 2;
    m_params.boxMax.z = boxDims.z / 2;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
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
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    }

    if (m_bUseOpenGL)
    {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));

        // fill color buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i=0; i<m_numParticles; i++)
        {
            float t = i / (float) m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

    sdkCreateTimer(&m_timer);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;

    if (m_bUseOpenGL)
    {
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL)
    {
    }
    else
    {
        dPos = (float *) m_cudaPosVBO;
    }

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
    }
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    
    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    printf("getArray unimplemented\n");
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                }
                else
                {
                }
            }
            break;

        case VELOCITY:
            break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    float w, h, d;
                    w = m_boxDims.x;
                    h = m_boxDims.y;
                    d = m_boxDims.z;
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius + m_params.boxMin.x + (w * frand() - w / 2)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius + m_params.boxMin.y + (h * frand() - h / 2)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius + m_params.boxMin.z + (d * frand() - d / 2)*jitter;
                    m_hPos[i*4+3] = 1.0f;

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float w, h, d;
                    w = m_boxDims.x;
                    h = m_boxDims.y;
                    d = m_boxDims.z;
                    m_hPos[p++] = w * frand() - w / 2;
                    m_hPos[p++] = h * frand() - h / 2;
                    m_hPos[p++] = d * frand() - d / 2;
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
            }
            break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;
    float w, h, d;
    w = m_boxDims.x;
    h = m_boxDims.y;
    d = m_boxDims.z;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (w * frand() - w / 2)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (h * frand() - h / 2)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (d * frand() - d / 2)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
