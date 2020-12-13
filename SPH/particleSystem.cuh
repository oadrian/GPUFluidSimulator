// header for c function definitions and cuda kernels
#include "particles_kernel.cuh"
extern "C"
{
	void cudaInit(int argc, char** argv);

	void allocateArray(void** devPtr, size_t size);
	void freeArray(void* devPtr);

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);
	void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);

	void threadSync();

	void copyArrayFromDevice(void *host, const void *device, size_t size);
	void copyArrayToDevice(void* device, const void* host, size_t size);

	void cudaComputeDensities(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params);
	void cudaComputeForces(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params);
	void cudaParticleCollisions(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint  dev_b_size, Grid_item* dev_B_prime, uint dev_B_prime_size, SimParams* params);

	void cudaConstructGridArray(Particle* dev_particles, uint dev_num_particles, Grid_item* dev_B, uint dev_b_size, Grid_item** dev_B_prime, uint* dev_B_prime_size, SimParams* params);
	void cudaIntegrate(float* gl_pos, float deltaTime, Particle* dev_particles, uint dev_num_particles, SimParams* params);
}