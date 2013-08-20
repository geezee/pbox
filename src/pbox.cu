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


typedef struct _wave_function {
    int n;             // the energy level of the wave function
    float energy;      // the energy of the particle at that level
    float probability; // the probability of getting this wave function (c^2_n)
} _wave_function;

typedef struct particle {
    float mass;            // the mass of the particle
    int waveCount;
} particle;



/**
 * Create a new wave
 *
 * @param _wave_function  the wave function to create
 * @param int             the energy level of the particle
 * @param float           the mass of the particle
 * @param float           the probability of the wave function (c^2_n)
*/
void _create_wave(_wave_function *wave, int n, float mass, float p) {
    wave->n = n;
    wave->energy = HBAR*HBAR * PI*PI * n*n / (mass * L*L);
    wave->probability = p;
}


/**
 * Given numbers, the function generates the next set of probabilities
 * based on these numbers.
 *
 * @param int            the numbers to generate the probabilities from
 * @param int            the length of the numbers array (should be the same
                         for the probabilities array);
 * @param _wave_function array of wave function to update their probabilities
*/
void next_probabilities(int *numbers, int N, _wave_function** wave) {
    int sum = 0;
    int carry = 1;
    for(int i=N-1;i>=0;i--) {
        numbers[i] +=  carry;
        carry       =  numbers[i] / 10;
        numbers[i] %=  10;
        sum        +=  numbers[i];
    }
    for(int i=0;i<N;i++) {
        wave[i]->probability = 1.0f*numbers[i] / sum;
    }
}

/**
 * Create a new particle
 * 
 * @param particle      the particle to create
 * @param int           the number of energy levels
 * @param float         the mass of the particle
*/
void create_particle(particle *p, int N, float mass) {
    p->mass = mass;
    p->waveCount = N;
    p->wave = (_wave_function **) malloc(N*sizeof(_wave_function));
    for(int i=0;i<N;i++) {
        p->wave[i] = (_wave_function *) malloc(sizeof(_wave_function));
        _create_wave(p->wave[i], i+1, mass, i == 0 /* 1 if i == 0, 0 otherwise */);
    }
}

/**
 * Get the probability of finding a particle in one dimension
 *
 * @param particle      the particle to find the probability of
 * @param float         the position in which to find the probability at
 * @return              the probability of finding the particle at the point x
*/
__device__
float cuda_probability_1d(_wave_function w, float x, float l) {
    int n   =  w.n;
    float s =  E/l - (1/(n*PI)) * cos(2*n*PI*x/l) * sin(n*PI / DIM);
    return s;
}

/**
 * Get the probability of finding a particle in two dimensions
 *
 * @param particle      the particle to find the probability of
 * @param float         the x-position in which to find the probability at
 * @param float         the y-position in which to find the probability at
 * @return              the probability of finding the particle at the point (x, y)
*/
__device__
float probability(_wave_function *w, int count, float x, float y, float l) {
    if(x > 0 && x < l && y > 0 && y < l) {
        float probability = 0;
        for(int i=0;i<count;i++) {
            return w[i];
            probability += (1.0/(l*l) *
                           cuda_probability_1d(w[i], x, l) *
                           cuda_probability_1d(w[i], y, l))
                           * (&w[i])->probability;
        }
        return probability;
    }
    return 0;
}

/**
 * CUDA kernel that finds the maximum of an array of numbers using reduction
 * and adds them to the partialMax list that will be later on processed
 *
 * @param float             the array of numbers
 * @param int               the length of the array
 * @param float             the partial list to add to
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
 * A function that given an array of numbers will return the maximum
 * of these numbers. I uses the cuda_max kernel.
 *
 * @param float             the array of numbers
 * @param int               the length of the array
 * @return float            the maximum of the numbers
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

__global__
void cuda_probabilities(_wave_function *w, int count, float *probabilities, float l, int dim) {
    int i      = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x  * gridDim.x;

    while(i < DIM*DIM) {
        float x = 1.0*(i/dim)/dim;
        float y = 1.0*(i%dim)/dim;

        probabilities[i] = probability(w, count, x, y, l);
        i += offset;
    }
}

/**
 * A function that finds the maximum probability in the box
 *
 * @param particle          the particle to find the highest probability of
 * @return float            the highest probability
*/
float max_probability(particle *p) {
    float *probabilities = (float*) malloc(DIM*DIM*sizeof(float));
    float *devProbabilities;
    _wave_function* devWaves;

    cudaMalloc((void**) &devWaves, p->waveCount*sizeof(_wave_function));
    cudaMalloc((void**) &devProbabilities, DIM*DIM*sizeof(float));

    cudaMemcpy(devWaves, p->wave, p->waveCount*sizeof(_wave_function), cudaMemcpyHostToDevice);
    cudaMemset(devProbabilities, 0, DIM*DIM*sizeof(float));

    cuda_probabilities<<<32, 256>>>(devWaves, p->waveCount, devProbabilities, L, DIM);
    cudaMemcpy(probabilities, devProbabilities, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<DIM*DIM;i++) {
        printf("%f\t", probabilities[i]);
    }
    printf("\n");

    cudaFree(devWaves);
    cudaFree(devProbabilities);
    free(probabilities);
    return 0;
}



int main(int argc, const char *argv[]) { 
    particle p;
    create_particle(&p, 10, 2.5);
    float max_proba = max_probability(&p);
    printf("The maximum probability is %.3f\n", max_proba);
    return 0;
}
