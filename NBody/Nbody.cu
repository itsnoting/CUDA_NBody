#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>

#define TIMESTEPS 100
#define PARTICLE_COUNT 10
#define DIM 2
#define G 6.673 * powf(10, -11)

using namespace std;


__device__ float distance(float x1, float y1, float x2, float y2)
{
	return sqrtf(powf((x1 - x2), 2) + powf((y1 - y2), 2));
}

__device__ float cube(float num){
	return powf(num, 3);
}

__global__ void myprint(){
	return;
}

__global__ void initialize(float *pos_x, float* pos_y, float* masses, float* velocities_x, float* velocities_y){
	curandState_t state;
	curand_init(0, 0, 0, &state);
	pos_x[threadIdx.x] = float(curand(&state) % 101 + (-50));
	pos_y[threadIdx.x] = float(curand(&state) % 101 + (-50));
	velocities_x[threadIdx.x] = float(curand(&state) % 11 + (-5)) / 1000.0;
	velocities_y[threadIdx.x] = float(curand(&state) % 11 + (-5)) / 1000.0;
	masses[threadIdx.x] = float(curand(&state) % 10000);
}
__global__ void updateVelocities(float *masses, float* velocities_x, float* velocities_y, float* pos_x, float* pos_y)
{
	int q = threadIdx.x;
	int k = threadIdx.y;
	if (q != k && q < PARTICLE_COUNT && k < PARTICLE_COUNT){
		velocities_x[q] -= (G * masses[q] * masses[k]) / cube(distance(pos_x[q], pos_y[q], pos_x[k], pos_y[k])) * (pos_x[q] - pos_x[k]);
		velocities_y[q] -= (G * masses[q] * masses[k]) / cube(distance(pos_x[q], pos_y[q], pos_x[k], pos_y[k])) * (pos_y[q] - pos_y[k]);
		return;
	}
}
__global__ void updatePositions(float *pos_x, float *pos_y, float* velocities_x, float* velocities_y){
	if (threadIdx.x < PARTICLE_COUNT){
		pos_x[threadIdx.x] += velocities_x[threadIdx.x];
		pos_y[threadIdx.x] += velocities_y[threadIdx.x];
	}
}

int main() {
	srand(1);
	dim3 threadsperblock(PARTICLE_COUNT, PARTICLE_COUNT);
	// Initialize the host variables
	float ih_pos_x[PARTICLE_COUNT];
	float ih_pos_y[PARTICLE_COUNT];
	float fh_pos_x[PARTICLE_COUNT];
	float fh_pos_y[PARTICLE_COUNT];

	float h_masses[PARTICLE_COUNT];
	
	// Initialize the device variables
	float d_pos_x[PARTICLE_COUNT];
	float d_pos_y[PARTICLE_COUNT];
	float d_masses[PARTICLE_COUNT];
	float d_velocities_x[PARTICLE_COUNT];
	float d_velocities_y[PARTICLE_COUNT];
	float d_distances_x[PARTICLE_COUNT];
	float d_distances_y[PARTICLE_COUNT];

	if ( 
		(cudaMalloc((void**)&d_pos_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_pos_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_masses, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) )

	{
		cout << "Error: Allocating memory" << endl;
		return 1;
	}
	initialize<<<PARTICLE_COUNT / 256 + 1, 256>>>(d_pos_x, d_pos_y, d_masses, d_velocities_x, d_velocities_y);
	cudaMemcpy(ih_pos_x, d_pos_x, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(ih_pos_y, d_pos_y, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_masses, h_masses, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);

	for (int i = 0; i < TIMESTEPS; ++i){
		updateVelocities<<<PARTICLE_COUNT / 256 + 1, threadsperblock>>>(d_pos_x, d_pos_y, d_masses, d_velocities_x, d_velocities_y);
		updatePositions<<<PARTICLE_COUNT / 256 + 1, 256 >>>(d_pos_x, d_pos_y, d_velocities_x, d_velocities_y);
	}
	cudaMemcpy(fh_pos_x, d_pos_x, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(fh_pos_y, d_pos_y, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	
	cout << "INITIAL POSITIONS" << endl;
	
	for (int i = 0; i < PARTICLE_COUNT; ++i){
		cout << i << ":\t" << h_masses[i] << "\t" << ih_pos_x[i] << ", " << ih_pos_y[i] << endl;
	}
	cout << "FINAL POSITIONS" << endl;
	for (int j = 0; j < PARTICLE_COUNT; ++j){
		cout << j << ":\t" << h_masses[j] << "\t" << ih_pos_x[j] << ", " << ih_pos_y[j] << endl;
	}
	getchar();


}