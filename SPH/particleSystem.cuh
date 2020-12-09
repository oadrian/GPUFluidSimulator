// header for c function definitions and cuda kernels

extern "C"
{
	void cudaInit(int argc, char** argv);

	void allocateArray(void** devPtr, int size);
	void freeArray(void* devPtr);

	void threadSync();

	void copyArrayFromDevice(void *host, const void *device, int size);
	void copyArrayToDevice(void* device, const void* host, int size);

	void cudaComputeDensities();
	void cudaComputeForces();
	void particleCollisions();
}