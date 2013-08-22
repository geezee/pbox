#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define GL_GLEXT_PROTOTYPES
#include "GL/glut.h"
#include "cuda_gl_interop.h"


#define PI   3.141592653589793
#define L    1.0          // the length of the box
#define DIM  512          // the dimensions of the window (should be 2^n)
#define E    (L/DIM)      // epsilon
#define HBAR 1.054        // without the e-34


#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b
#define ERROR(msg) do {                                               \
            fprintf(stderr, "%s(%d): %s\n", __FILE__, __LINE__, msg); \
            exit(0);                                                  \
        } while(0);


GLuint buffer;
cudaGraphicsResource *resource;


/* The struct that describes a particle */
typedef struct particle {
    float mass;            // mass of the particle
    int energy_levels;     // the number of energy levels
    float *probabilities;  // the probabilitis of every energy level
} particle;


/**
 * CUDA kernel that finds the maximum in buckets of numbers
 * and fills it in the partialMax array using the reduction method
 *
 * @param float*            the numbers to find the maximum of
 * @param int               the length of the array (param 1)
 * @param float*            list of buckets containing the maximums
*/
__global__
void cuda_max(float *numbers, int N, float *partialMax) {
    extern __shared__ float cache[];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float m = 0;
    while(tid < N) {
        m = MAX(m, numbers[tid]);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = m;
    __syncthreads();

    int i = blockDim.x/2;
    while(i != 0) {
        if(cacheIndex < i)
            cache[cacheIndex] = MAX(cache[cacheIndex], cache[cacheIndex + i]);
        __syncthreads();

        i /= 2;
    }

    if(cacheIndex == 0)
        partialMax[blockIdx.x] = cache[0];
}

/**
 * Function that finds the maximum of an array of numbers using the
 * cuda_max function
 *
 * @param float*            the array of numbers to find the maximum of
 * @param int               the length of the array
 * @return float            the maximum of the array
*/
float max(float *numbers, int N) {
    int tpb = 256; // threads per block
    int bpg = MIN(32, (N+tpb-1)/tpb); // blocks per grid

    float *partialMax = (float *) malloc(bpg*sizeof(float));
    float *devNumbers, *devPartialMax;

    cudaMalloc((void**) &devNumbers, N*sizeof(float));
    cudaMalloc((void**) &devPartialMax, bpg*sizeof(float));
    
    cudaMemcpy(devNumbers, numbers, N*sizeof(float), cudaMemcpyHostToDevice);
    cuda_max<<<tpb, bpg, bpg*sizeof(float)>>>(devNumbers, N, devPartialMax);
    cudaMemcpy(partialMax, devPartialMax, bpg*sizeof(float), cudaMemcpyDeviceToHost);

    float m = partialMax[0];
    for(int i=1;i<bpg;i++)
        m = MAX(m, partialMax[i]);

    cudaFree(devNumbers);
    cudaFree(devPartialMax);
    free(partialMax);
    return m;
}

/**
 * Given an array of numbers, the function finds a next set
 * of probabilities
 *
 * @param int*                the numbers to act upon
 * @param int                 the length of the array
 * @param float*              where to store the next set of probabilities
*/
void next_probabilities(int *numbers, int N, float* probabilities) {
    int sum = 0;
    int carry = 1;
    for(int i=0;i<N;i++) {
        numbers[i] +=  carry;
        carry       =  numbers[i] / 10;
        numbers[i] %=  10;
        sum        +=  numbers[i];
    }
    for(int i=0;i<N;i++) {
        probabilities[i] = 1.0f*numbers[i] / sum;
    }
}

/**
 * A function that creates a new particle
 *
 * @param particle*           the particle to create
 * @param int                 the number of energy levels the particle can have
 * @param float               the mass of the particle
*/
void create_particle(particle *p, int N, float mass) {
    p->mass = mass;
    p->energy_levels = N;
    p->probabilities = (float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++)
        p->probabilities[i] = (i == 0); // 1 if i == 0, 0 otherwise
                                        // so sum(probabilities) = 1
}

/**
 * CUDA device function that finds the probability of finding the
 * particle in one dimension at the position x at the energy level n
 *
 * @param int                  the energy level of the particle
 * @param float                the position of the particle
 * @return float               the probability
*/
__device__
float cuda_probability_1d_device(int n, float x) {
    float s =  E/L - (1/(n*PI)) * cos(2*n*PI*x/L) * sin(n*PI / DIM);
    return s;
}

/**
 * CUDA device function that finds the probabiltiy of finding the 
 * particle at a fixed position given a set of probabilities, the
 * number of energy levels and the position
 *
 * @param float*                the probability of each energy level
 * @param int                   the number of energy levels
 * @param float                 the x-coordinate of the particle
 * @param float                 the y-coordinate of the particle
 * @return float                the probability
*/
__device__
float cuda_probability_2d_device(float *probability, int n, float x, float y) {
    float s = 0;
    for(int i=0;i<n;i++) {
        s += (1.0/(L*L) * cuda_probability_1d_device(i+1, x) *
                          cuda_probability_1d_device(i+1, y)) * probability[i];
    }
    return s;
}

/**
 * CUDA kernel that maps the coordinate array to a probability array
 *
 * @param float*                 the probability set of each energy level
 * @param int                    the number of energy levels
 * @param float*                 the array to map to
*/
__global__ 
void cuda_probability_to_map(float *probabilities, int n, float *map) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    while(i < DIM * DIM) {
        float x = 1.0*L*((int) (i%DIM))/DIM;
        float y = 1.0*L*((int) (i/DIM))/DIM;

        if(x > 0 && x < L && y > 0 && y < L)
            map[i] = cuda_probability_2d_device(probabilities, n, x, y);
        else
            map[i] = 0;
        i += offset;
    }
}

/**
 * CUDA kernel to find the probability of finding the particle at a certain
 * position
 *
 * @param float*                  the probability set of each energy level
 * @param int                     the number of energy levels
 * @param float                   the x-coordinate of the particle
 * @param float                   the y-coordinate of the particle
 * @param float*                  used to write the probability to
*/
__global__
void cuda_probability(float *p, int N, float x, float y, float *probability) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    if(i == 0)
        probability[0] = 0;
    __syncthreads();
    
    while(i < N) {
        if(x > 0 && x < L && y > 0 && y < L) {
            atomicAdd(&probability[0],  ( 1.0/(L*L) *
                                          cuda_probability_1d_device(i+1, x) *
                                          cuda_probability_1d_device(i+1, y)
                                        ) * p[i]);
        } else {
            probability[0] = 0;
        }
        i += offset;
    }
}

/**
 * A function that finds the probability of finding the particle at a certain position
 * @param particle* p               the particle
 * @param float                     the x-coordinate of the particle
 * @param float                     the y-coordinate of the particle
 * @return                          the probability
*/
float probability(particle *p, float x, float y) {
    float *devProbabilities, *devProbability, *probability;

    cudaMalloc((void**) &devProbabilities, p->energy_levels*sizeof(float));
    cudaMalloc((void**) &devProbability, sizeof(float));
    probability = (float*) malloc(sizeof(float));

    cudaMemcpy(devProbabilities, p->probabilities, p->energy_levels*sizeof(float), cudaMemcpyHostToDevice);
    cuda_probability<<<1, p->energy_levels>>>(devProbabilities, p->energy_levels, x, y, devProbability);
    cudaMemcpy(probability, devProbability, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devProbabilities);
    cudaFree(devProbability);
    return probability[0];
}

/**
 * A function that finds the maximum probability in the space of finding the particle
 * This function is used to map the probabilities [0, max] |-> [0, 255]
 * 
 * @param particle*                  the particle
 * @return                           the highest probability
*/
float max_probability(particle *p) {
     // Before analyzing the mathematical equation I would brute-force the problem
     // by looping over all the values and picking the maximum
    float *devProbabilities, *devMap;
    float *map;

    map = (float*) malloc(DIM * DIM * sizeof(float));
    cudaMalloc((void**) &devProbabilities, p->energy_levels * sizeof(float));
    cudaMalloc((void**) &devMap, DIM * DIM * sizeof(float));
    
    cudaMemcpy(devProbabilities, p->probabilities, p->energy_levels * sizeof(float), cudaMemcpyHostToDevice);
    cuda_probability_to_map<<<32, 256>>>(devProbabilities, p->energy_levels, devMap);
    cudaMemcpy(map, devMap, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    return max(map, DIM*DIM);

    float probability = E/L, p_1d = 0.0;
    int n;
    for(int i=0;i<10;i++) {
        n = i+1;
        p_1d = E/L + 1.0/(n*PI) * sin(n*PI*E);
        probability += p->probabilities[i] * p_1d * p_1d;
    }

    return probability;
}

/**
 * A function that finds the energy of the particle at a precise energy level
 *
 * @param float                     the mass of the particle
 * @param int                       the energy level
*/
__device__
float energy(float mass, int n) {
    return (HBAR * HBAR * PI * PI * n * n) / (mass * L * L);
}

/**
 * A function that finds the highest energy the particle can reach
 * This function is useful for color mapping
 *
 * @param float                     the mass of the particle
 * @return float                    the highest energy
*/
__device__
float highest_energy(float mass, int n) {
    return energy(mass, n);
}

int main(int argc, char *argv[]) { 
    return 0;
}
