// implementation of kernels

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include "particleSystem.cuh"


__device__ void computePressureIdeal(Particle* p) {
	p->pressure = fmaxf(0.f, GAS_CONSTANT * (p->density - REST_DENS));
}

__device__ void computePressureTait(Particle* p) {
	const float cs = 88.5f;
	const float gamma = 7.f;
	const float B = (REST_DENS * cs * cs) / gamma;
	float quot = p->density / REST_DENS;
	float quot_exp_gamma = quot * quot * quot * quot * quot * quot * quot;
	p->pressure = B * ((quot_exp_gamma)-1.f);
}

__device__ void computeDensity(Particle* pi, const Particle* pj) {
	//printf("hi from computeDensity\n");
	const float POLY6 = 315.f / (65.f * PI_F * powf(m_H, 9.f));
	Vector3f rij = pi->position - pj->position;
	float r2 = rij.squaredNorm();
	Vector3f t = { 1.f, 2.f, 3.f }, d = { 2.f, 2.f, 2.f };
	if (r2 < HSQ) {
		pi->density += pj->mass * POLY6 * powf(HSQ - r2, 3.f);
	}
}

__device__ void computeForce(Particle* pi, const Particle* pj) {
	//printf("hi from computeForce\n");
	const float SPIKY_GRAD = -45.f / (PI_F * powf(m_H, 6.f));
	const float VISC_LAP = 45.f / (PI_F * powf(m_H, 6.f));
	Vector3f rij = pi->position - pj->position;
	float r = rij.norm();
	Vector3f t = { 1.f, 2.f, 3.f }, d = { 2.f, 2.f, 2.f };
	if (r < m_H) {
		pi->force_press += -rij.normalized() * pj->mass * (pi->pressure + pj->pressure) / (2.f * pj->density) * SPIKY_GRAD * powf(m_H - r, 2.f);
		pi->force_visc += VISC * pj->mass * (pj->velocity - pi->velocity) / pj->density * VISC_LAP * (m_H - r);
	}
}

__device__ void computeCollision(Particle* pi, const Particle* pj) {
	//printf("hi from computeCollision\n");
	if (pi->index == pj->index) return;
	Vector3f vij, rij;
	float dij;
	vij = pi->velocity - pj->velocity;
	rij = pi->position - pj->position;
	dij = rij.norm();
	Vector3f t = { 1.f, 2.f, 3.f }, d = { 2.f, 2.f, 2.f };
	if (dij <= COLLISION_PARAM * 2 * pi->radius && rij.dot(vij) < 0) {
		pi->delta_velocity += (pj->mass * (1.f + RESTITUTION)) * (rij.dot(vij) / (dij * dij)) * rij;
		pi->collision_count++;
	}
}

// coord components must be 10 bits
__device__ uint coord2zIndex(Vector3i coord) {
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

__device__ uint get_Z_index(Particle p, SimParams *params) {
    // find the section of the grid the particle is in
    float posX = p.position.x() - params->boxMin.x;
    float posY = p.position.y() - params->boxMin.y;
    float posZ = p.position.z() - params->boxMin.z;
    Vector3i coord;
    coord.x() = floor((posX / params->boxDims.x) * params->gridDim);
    coord.y() = floor((posY / params->boxDims.y) * params->gridDim);
    coord.z() = floor((posZ / params->boxDims.z) * params->gridDim);
    return coord2zIndex(coord);
}

// 30  bit number -> 10 bit number
__device__ uint cudaCollapseEvery3(uint x) {
    uint res = 0;
    for (int i = 0; i < 10; i++) {
        uint b = ((x >> i * 3) & 1);
        res = res | (b << i);
    }
    return res;
}

__device__ dim3 zIndex2coord(uint z_index) {
    uint x = cudaCollapseEvery3(z_index);
    uint y = cudaCollapseEvery3(z_index >> 1);
    uint z = cudaCollapseEvery3(z_index >> 2);
		dim3 res;
		res.x = x;
		res.y = y;
		res.z = z;
    return res;
}

// return if the given coordinates are in the bounds of the sim grid
__device__ int inGrid(int x, int y, int z, SimParams *params) {
	int gridMax = params->gridDim;
	return (x < gridMax) && (y < gridMax) && (z < gridMax) && (x >= 0) && (y >= 0) && (z >= 0);
}

__global__ void kernelComputeDensities(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, Grid_item* dev_B_prime, SimParams* params) {
	__shared__ Particle batch[GRID_COMPACT_WIDTH];
	__shared__ int total_cnt;
	__shared__ int batch_cnt;
	uint blockid = blockIdx.x;
	if (dev_B_prime[blockid].nParticles == 0) return;
	bool valid = threadIdx.x < dev_B_prime[blockid].nParticles;
	uint particleid = (valid) ? dev_B_prime[blockid].start + threadIdx.x : dev_B_prime[blockid].start;
	Particle *pi = &dev_particles[particleid];
	if(valid) pi->density = 0.f;

	// for each neighboring grid block
	dim3 coords = zIndex2coord(pi->zindex);
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			for (int dz = -1; dz < 2; dz++) {
				int neighborx = coords.x + dx;
				int neighbory = coords.y + dy;
				int neighborz = coords.z + dz;
				// check coordinates are valid
				if (!inGrid(neighborx, neighbory, neighborz, params)) {
					continue;
				}
				// get the zindex of the block
				int blockZIndex = coord2zIndex(Vector3i(neighborx, neighbory, neighborz));
				Grid_item neighbor_block = dev_B[blockZIndex];
				int start = neighbor_block.start;
				int nParticles = neighbor_block.nParticles;
				total_cnt = 0;
				batch_cnt = 0;
				__syncthreads();

				while (total_cnt < nParticles) {
					// in batches, copy particles into the batch array for processing
					int toCopyDex = total_cnt + threadIdx.x;
					batch_cnt = 0;
					__syncthreads();
					if (toCopyDex < nParticles) {
						batch[threadIdx.x] = dev_particles[start + toCopyDex];
						atomicAdd(&batch_cnt, 1); // count the particle as copied
					}
					__syncthreads();
					// each active thread computes values for itself from its neighbors
					for (int j = 0; j < batch_cnt; j++) {
						Particle *pj = &batch[j];
						if (valid) computeDensity(pi, pj);
					}
					if(threadIdx.x == 0) total_cnt += batch_cnt;
					__syncthreads();
				}
			}
		}
	}
	__syncthreads();
	if (valid) computePressureIdeal(pi);
}

__global__ void kernelComputeForces(Particle *dev_particles, uint dev_num_particles, Grid_item *dev_B, Grid_item *dev_B_prime, SimParams *params) {
	__shared__ Particle batch[GRID_COMPACT_WIDTH];
	__shared__ int total_cnt;
	__shared__ int batch_cnt;
	uint blockid = blockIdx.x;
	if (dev_B_prime[blockid].nParticles == 0) return;
	bool valid = threadIdx.x < dev_B_prime[blockid].nParticles;
	uint particleid = (valid) ? dev_B_prime[blockid].start + threadIdx.x : dev_B_prime[blockid].start;
	Particle* pi = &dev_particles[particleid];
	if (valid) pi->force_press = { 0.f, 0.f, 0.f };
	if (valid) pi->force_visc = { 0.f, 0.f, 0.f };

	// for each neighboring grid block
	dim3 coords = zIndex2coord(pi->zindex);
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			for (int dz = -1; dz < 2; dz++) {
				int neighborx = coords.x + dx;
				int neighbory = coords.y + dy;
				int neighborz = coords.z + dz;
				// check coordinates are valid
				if (!inGrid(neighborx, neighbory, neighborz, params)) {
					continue;
				}
				// get the zindex of the block
				int blockZIndex = coord2zIndex(Vector3i(neighborx, neighbory, neighborz));
				Grid_item neighbor_block = dev_B[blockZIndex];
				int start = neighbor_block.start;
				int nParticles = neighbor_block.nParticles;
				total_cnt = 0;
				batch_cnt = 0;
				__syncthreads();

				while (total_cnt < nParticles) {
					// in batches, copy particles into the batch array for processing
					int toCopyDex = total_cnt + threadIdx.x;
					batch_cnt = 0;
					__syncthreads();
					if (toCopyDex < nParticles) {
						batch[threadIdx.x] = dev_particles[start + toCopyDex];
						atomicAdd(&batch_cnt, 1); // count the particle as copied
					}
					__syncthreads();
					// each active thread computes values for itself from its neighbors
					for (int j = 0; j < batch_cnt; j++) {
						Particle* pj = &batch[j];
						if (valid) computeForce(pi, pj);
					}
					if (threadIdx.x == 0) total_cnt += batch_cnt;
					__syncthreads();
				}
			}
		}
	}
}

__global__ void kernelComputeCollisions(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, Grid_item* dev_B_prime, SimParams* params) {
	__shared__ Particle batch[GRID_COMPACT_WIDTH];
	__shared__ int total_cnt;
	__shared__ int batch_cnt;
	uint blockid = blockIdx.x;
	if (dev_B_prime[blockid].nParticles == 0) return;
	bool valid = threadIdx.x < dev_B_prime[blockid].nParticles;
	uint particleid = (valid) ? dev_B_prime[blockid].start + threadIdx.x : dev_B_prime[blockid].start;
	Particle* pi = &dev_particles[particleid];
	if (valid) pi->delta_velocity = { 0.f, 0.f, 0.f };
	if (valid) pi->collision_count = 0;

	// for each neighboring grid block
	dim3 coords = zIndex2coord(pi->zindex);
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			for (int dz = -1; dz < 2; dz++) {
				int neighborx = coords.x + dx;
				int neighbory = coords.y + dy;
				int neighborz = coords.z + dz;
				// check coordinates are valid
				if (!inGrid(neighborx, neighbory, neighborz, params)) {
					continue;
				}
				// get the zindex of the block
				int blockZIndex = coord2zIndex(Vector3i(neighborx, neighbory, neighborz));
				Grid_item neighbor_block = dev_B[blockZIndex];
				int start = neighbor_block.start;
				int nParticles = neighbor_block.nParticles;
				total_cnt = 0;
				batch_cnt = 0;
				__syncthreads();

				while (total_cnt < nParticles) {
					// in batches, copy particles into the batch array for processing
					int toCopyDex = total_cnt + threadIdx.x;
					batch_cnt = 0;
					__syncthreads();
					if (toCopyDex < nParticles) {
						batch[threadIdx.x] = dev_particles[start + toCopyDex];
						atomicAdd(&batch_cnt, 1); // count the particle as copied
					}
					__syncthreads();
					// each active thread computes values for itself from its neighbors
					for (int j = 0; j < batch_cnt; j++) {
						Particle* pj = &batch[j];
						if (valid) computeCollision(pi, pj);
					}
					if (threadIdx.x == 0) total_cnt += batch_cnt;
					__syncthreads();
				}
			}
		}
	}
	if (valid) pi->delta_velocity = -pi->delta_velocity / (pi->mass * (1 + pi->collision_count));
}

__global__ void kernelGetZIndex(Particle* dev_particles, uint dev_num_particles, SimParams* params) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dev_num_particles) return; // map one cuda thread per particle

	Particle p = dev_particles[index];
	p.zindex = get_Z_index(p, params);
	
}

__global__ void kernelConstructBGrid(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint dev_b_size, SimParams* params) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dev_num_particles) return; // one cuda thread per particle again
	Particle p = dev_particles[index];
	unsigned long long zind = p.zindex;
	// continue taking the min of particle indices with the same z index to find the starting index
	atomicMin(&(dev_B[zind].start), index);
	// atomically increment the particle count
	atomicAdd(&(dev_B[zind].nParticles), 1);
}

// returns the number of blocks in B' that will exist below a certain particle
__device__ int numBlocksBelow(int zindex, Grid_item* dev_B) {
	int blocks = 0;
	for (int i = 0; i < zindex; i++) {
		int particlesInBlock = dev_B[i].nParticles;
		blocks += ceil(particlesInBlock / GRID_COMPACT_WIDTH);
	}
	return blocks;
}

__global__ void kernelConstructBPrimeGrid(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dev_num_particles) return;
	Particle p = dev_particles[index];
	unsigned long long zind = p.zindex;
	int numParticlesInBlock = dev_B[zind].nParticles;
	int particleStartingIndex = dev_B[zind].start;
	int localIndex = index - particleStartingIndex;
	int b_prime_dex = numBlocksBelow(zind, dev_B) + floor(localIndex / GRID_COMPACT_WIDTH);
	// continue taking the min of particle indices with the same B' index to find the starting index
	atomicMin(&(dev_B_prime[b_prime_dex].start), index);
	// atomically increment the particle count
	atomicAdd(&(dev_B_prime[b_prime_dex].nParticles), 1);
}

extern "C" {
	void cudaInit(int argc, char** argv) {
		int devID;

		devID = findCudaDevice(argc, (const char**)argv);

		if (devID < 0) {
			printf("No CUDA devices found, exiting\n");
			exit(EXIT_SUCCESS);
		}
	}

	void allocateArray(void** devPtr, size_t size) {
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void* devPtr) {
		checkCudaErrors(cudaFree(devPtr));
	}

	void threadSync() {
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void copyArrayFromDevice(void* host, const void* device, size_t size) {
		checkCudaErrors(cudaMemcpy((char*)host, device, size, cudaMemcpyDeviceToHost));
	}

	void copyArrayToDevice(void* device, const void* host, size_t size) {
		checkCudaErrors(cudaMemcpy((char*)device, host, size, cudaMemcpyHostToDevice));
	}

	void cudaComputeDensities(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params) {
		kernelComputeDensities <<<dev_B_prime_size, GRID_COMPACT_WIDTH>>>(dev_particles, dev_num_particles, dev_B, dev_B_prime, params);
	}

	void cudaComputeForces(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params) {
		kernelComputeForces <<<dev_B_prime_size, GRID_COMPACT_WIDTH>>>(dev_particles, dev_num_particles, dev_B, dev_B_prime, params);
	}

	void cudaParticleCollisions(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params) {
		kernelComputeCollisions <<<dev_B_prime_size, GRID_COMPACT_WIDTH>>>(dev_particles, dev_num_particles, dev_B, dev_B_prime, params);
	}

	void cudaConstructGridArray(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params) {
		int blocks = ceil(dev_num_particles / GRID_COMPACT_WIDTH);
		// set particles' z indices
		kernelGetZIndex <<<blocks, GRID_COMPACT_WIDTH>>> (dev_particles, dev_num_particles, params);
		// sort according to z index
		thrust::sort(dev_particles, dev_particles + dev_num_particles, particle_cmp());
		// clear the previous grid arrays
		cudaMemset(dev_B, 0, dev_b_size * sizeof(Grid_item));
		cudaMemset(dev_B_prime, 0, dev_B_prime_size * sizeof(Grid_item));
		// set the B grid
		kernelConstructBGrid <<<blocks, GRID_COMPACT_WIDTH>>> (dev_particles, dev_num_particles, dev_B, dev_b_size, params);
		// set up the B' grid

	}

	void cudaIntegrate(float deltaTime, Particle* dev_particles, uint dev_num_particles, SimParams* params) {
		return;
	}
}
