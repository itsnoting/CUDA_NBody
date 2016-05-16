#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>

#define TIMESTEPS 100
#define PARTICLE_COUNT 10
#define DIM 2
#define G 6.673 * pow(10, -11)

using namespace std;


__device__ float distance(float x1, float y1, float x2, float y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
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
	velocities_x[threadIdx.x] -= (G * masses[threadIdx.x] * masses[threadIdx.y]) / 
		pow(distance(pos_x[threadIdx.x], pos_y[threadIdx.x], pos_x[threadIdx.y], pos_y[threadIdx.y]), 3) * 
		(pos_x[threadIdx.x] - pos_x[threadIdx.y]);
	velocities_y[threadIdx.x] -= (G * masses[threadIdx.x] * masses[threadIdx.y]) / 
		pow(distance(pos_x[threadIdx.x], pos_y[threadIdx.x], pos_x[threadIdx.y], pos_y[threadIdx.y]), 3) * 
		(pos_y[threadIdx.x] - pos_y[threadIdx.y]);

}
__global__ void updatePositions(float *pos_x, float *pos_y, float* masses, float* velocities_x, float* velocities_y){
	
}

int main() {
	srand(1);
	dim3 threadsperblock(PARTICLE_COUNT, DIM);
	// Initialize the host variables
	float h_pos_x[PARTICLE_COUNT]= {};
	float h_pos_y[PARTICLE_COUNT] = {};
	float h_masses[PARTICLE_COUNT] = {};
	float h_velocities_x[PARTICLE_COUNT] = {};
	float h_velocities_y[PARTICLE_COUNT] = {};
	float h_distances_x[PARTICLE_COUNT] = {};
	float h_distances_y[PARTICLE_COUNT] = {};

	// Initialize the device variables
	float d_pos_x[PARTICLE_COUNT] = {};
	float d_pos_y[PARTICLE_COUNT] = {};
	float d_masses[PARTICLE_COUNT] = {};
	float d_velocities_x[PARTICLE_COUNT] = {};
	float d_velocities_y[PARTICLE_COUNT] = {};
	float d_distances_x[PARTICLE_COUNT] = {};
	float d_distances_y[PARTICLE_COUNT] = {};

	if ( 
		(cudaMalloc((void**)&d_pos_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_pos_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_masses, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_x, sizeof(float) * PARTICLE_COUNT * DIM) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_y, sizeof(float) * PARTICLE_COUNT * DIM) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_x, sizeof(float) * PARTICLE_COUNT * DIM) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_y, sizeof(float) * PARTICLE_COUNT * DIM) != cudaSuccess) )

	{
		cout << "Error: Allocating memory" << endl;
		return 1;
	}
	initialize << < PARTICLE_COUNT / 256 + 1, 256 >> >(d_pos_x, d_pos_y, d_masses, d_velocities_x, d_velocities_y);
	for (int i = 0; i < TIMESTEPS; ++i){
		updatePositions << <PARTICLE_COUNT / 256 + 1, 256 >> >(d_pos_x, d_pos_y, d_masses, d_velocities_x, d_velocities_y);
		continue;
	}

}