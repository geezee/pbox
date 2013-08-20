#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define PI   3.1415
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


typedef struct particle {
    float mass;
    int energy_levels;
    float *probabilities;
} particle;


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

void next_probabilities(int *numbers, int N, float* probabilities) {
    int sum = 0;
    int carry = 1;
    for(int i=N-1;i>=0;i--) {
        numbers[i] +=  carry;
        carry       =  numbers[i] / 10;
        numbers[i] %=  10;
        sum        +=  numbers[i];
    }
    for(int i=0;i<N;i++) {
        probabilities[i] = 1.0f*numbers[i] / sum;
    }
}

void create_particle(particle *p, int N, float mass) {
    p->mass = mass;
    p->energy_levels = N;
    p->probabilities = (float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++)
        p->probabilities[i] = (i == 0); // 1 if i == 0, 0 otherwise
                                        // so sum(probabilities) = 1
}

__device__
float cuda_probability_1d_device(int n, float x) {
    float s =  E/L - (1/(n*PI)) * cos(2*n*PI*x/L) * sin(n*PI / DIM);
    return s;
}

__device__
float cuda_probability_2d_device(float *probability, int n, float x, float y) {
    float s = 0;
    for(int i=0;i<n;i++) {
        s += (1.0/(L*L) * cuda_probability_1d_device(i+1, x) *
                          cuda_probability_1d_device(i+1, y)) * probability[i];
    }
    return s;
}

__global__ 
void cuda_max_probability(float *probabilities, int n, float *map) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    while(i < DIM * DIM) {
        float x = 1.0*((int) (i/DIM))/DIM;
        float y = 1.0*((int) (i%DIM))/DIM;

        if(x > 0 && x < L && y > 0 && y < L)
            map[i] = cuda_probability_2d_device(probabilities, n, x, y);
        else
            map[i] = 0;
        i += offset;
    }
}

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

float max_probability(particle *p) {
    float *devProbabilities, *devMap;
    float *map;

    map = (float*) malloc(DIM * DIM * sizeof(float));
    cudaMalloc((void**) &devProbabilities, p->energy_levels * sizeof(float));
    cudaMalloc((void**) &devMap, DIM * DIM * sizeof(float));
    
    cudaMemcpy(devProbabilities, p->probabilities, p->energy_levels * sizeof(float), cudaMemcpyHostToDevice);
    cuda_max_probability<<<32, 256>>>(devProbabilities, p->energy_levels, devMap);
    cudaMemcpy(map, devMap, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    return max(map, DIM*DIM);
}

int main(int argc, const char *argv[]) { 
    particle p;
    float elapsed;
    int N = 10;

    create_particle(&p, N, 2.5);
    int *numbers = (int*) malloc(N*sizeof(int));
    for(int i=0;i<N;i++) numbers[i] = i==0;

    next_probabilities(numbers, N, p.probabilities);
    for(int i=0;i<N;i++)
        printf("Probability of wave %d is %f\n", i+1, p.probabilities[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    printf("The maximum is %f\n", max_probability(&p));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("It took %fms to find the maximum\n", elapsed);

    float x = 0.5, y = 0.5;
    printf("The probability at (%.1f, %.1f) is %f\n", x, y, probability(&p, x, y));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
