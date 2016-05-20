#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <ctime>

// Number of timesteps for the body movement simulation
#define TIMESTEPS 1000

// Number of bodies in the given domain
#define PARTICLE_COUNT 10

// Dimentions of the given domain
#define DIM 2

// Gravitational constant
#define G 6.673 * powf(10, -11)

using namespace std;


__device__ float distance(float x1, float y1, float x2, float y2)
{
	// Distance formula
	return sqrtf(powf((x1 - x2), 2) + powf((y1 - y2), 2));
}

__device__ float cube(float num){
	// Cube num
	return powf(num, 3);
}

__global__ void updateVelocities(float *masses, float* velocities_x, float* velocities_y, float* pos_x, float* pos_y)
{
	// Using threadIdx x as particle q's index
	int q = threadIdx.x;
	// Using threadIdx y as particle k's index
	int k = threadIdx.y;

	// Using a device function, I find the distance between particle q and particle k
	float qkdistance = distance(pos_x[q], pos_y[q], pos_x[k], pos_y[k]);

	// cube the distance for the formula to find the force
	float dist_cubed = cube(qkdistance);

	float x_diff = pos_x[q] - pos_x[k];
	float y_diff = pos_y[q] - pos_y[k];
	
	// After calculating the forces on the x and y axes, I update the velocities.
	if (q != k && q < PARTICLE_COUNT && k < PARTICLE_COUNT){
		velocities_x[q] -= (G * masses[q] * masses[k]) / dist_cubed * x_diff;
		velocities_y[q] -= (G * masses[q] * masses[k]) / dist_cubed * y_diff;
		return;
	}
}
__global__ void updatePositions(float* velocities_x, float* velocities_y, float *pos_x, float *pos_y){

	// After updating the velocities, I use the velocities to find the new positions.
	if (threadIdx.x < PARTICLE_COUNT){
		pos_x[threadIdx.x] += velocities_x[threadIdx.x];
		pos_y[threadIdx.x] += velocities_y[threadIdx.x];
	}
}

int main() {

	dim3 threadsperblock(PARTICLE_COUNT, PARTICLE_COUNT);
	// Initialize the host variables
	float* h_pos_x = (float*)malloc(PARTICLE_COUNT * sizeof(float));
	float* h_pos_y = (float*)malloc(PARTICLE_COUNT * sizeof(float));
	float* h_velocities_x = (float*)malloc(PARTICLE_COUNT * sizeof(float));
	float* h_velocities_y = (float*)malloc(PARTICLE_COUNT * sizeof(float));
	float* h_masses = (float*)malloc(PARTICLE_COUNT * sizeof(float));

	// Initialize the device variables
	float* d_pos_x;
	float* d_pos_y;
	float* d_masses;
	float* d_velocities_x;
	float* d_velocities_y;
	float* d_distances_x;
	float* d_distances_y;

	//Checks if the device was able to allocate memory for the device variables
	if ( 
		(cudaMalloc((void**)&d_pos_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_pos_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_masses, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_velocities_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_x, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) ||
		(cudaMalloc((void**)&d_distances_y, sizeof(float) * PARTICLE_COUNT) != cudaSuccess) )

	{
		// If unsuccessful, print error message and return an exit code 1
		cout << "Error: Allocating memory" << endl;
		return 1;
	}


	// Initialize the particles positions, mass, and velocities.
	srand(1);
	for (int i = 0; i < PARTICLE_COUNT; ++i){
		h_pos_x[i] = (rand() % 101) - 50;
		h_pos_y[i] = (rand() % 101) - 50;
		h_masses[i] = rand() % 1000;
		h_velocities_x[i] = 0;
		h_velocities_y[i] = 0;

	}
	

	// After initializing, copy the variables to the device to perform calculations.
	cudaMemcpy(d_pos_x, h_pos_x, sizeof(float) * PARTICLE_COUNT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_y, h_pos_y, sizeof(float) * PARTICLE_COUNT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_masses, h_masses, sizeof(float) * PARTICLE_COUNT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities_x, h_velocities_x, sizeof(float) * PARTICLE_COUNT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities_y, h_velocities_y, sizeof(float) * PARTICLE_COUNT, cudaMemcpyHostToDevice);

	// Printing initial positions
	cout << "INITIAL POSITIONS" << endl;

	for (int i = 0; i < PARTICLE_COUNT; ++i){
		cout << i << ":\t" << h_masses[i] << "\t\t\t" << h_pos_x[i] << ", " << h_pos_y[i] << endl;
	}

	// Call the kernel functions to calculate the end point.
	for (int i = 0; i < TIMESTEPS; ++i){
		updateVelocities << <PARTICLE_COUNT / 256 + 1, threadsperblock >> >(d_masses, d_velocities_x, d_velocities_y, d_pos_x, d_pos_y);
		updatePositions << <PARTICLE_COUNT / 256 + 1, 256 >> >(d_velocities_x, d_velocities_y, d_pos_x, d_pos_y);
	}

	// Copy the results back to the host position variables
	cudaMemcpy(h_pos_x, d_pos_x, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos_y, d_pos_y, sizeof(float) * PARTICLE_COUNT, cudaMemcpyDeviceToHost);
	

	// Printing the final positions after calculations
	cout << "FINAL POSITIONS" << endl;
	for (int j = 0; j < PARTICLE_COUNT; ++j){
		cout << j << ":\t" << h_masses[j] << "\t\t\t" << h_pos_x[j] << ", " << h_pos_y[j] << endl;
	}


	// Clean up
	cudaFree(d_pos_x);
	cudaFree(d_pos_y);
	cudaFree(d_distances_x);
	cudaFree(d_distances_y);
	cudaFree(d_velocities_x);
	cudaFree(d_velocities_y);
	free(h_masses);
	free(h_pos_x);
	free(h_pos_y);
	free(h_velocities_x);
	free(h_velocities_y);

	getchar();


}