#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define PI   3.1415
#define L    1.0          // the length of the box
#define DIM  500          // the dimensions of the window
#define E    (L/DIM)      // epsilon
#define HBAR 1.054        // without the e-34

#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b
#define ERROR(msg) do { fprintf(stderr, "%s(%d): %s\n", __FILE__, __LINE__, msg); exit(0); } while(0);


typedef struct _wave_function {
    int n;             // the energy level of the wave function
    float energy;      // the energy of the particle at that level
    float probability; // the probability of getting this wave function (c^2_n)
} _wave_function;

typedef struct particle {
    float mass;            // the mass of the particle
    _wave_function **wave; // array of wave functions for the particle 
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
 * Get the probability of finding a particle in one dimension
 *
 * @param particle      the particle to find the probability of
 * @param float         the position in which to find the probability at
 * @return              the probability of finding the particle at the point x
*/
float _probability_1d(_wave_function *w, float x) {
    int n   =  w->n;
    float s =  E/L - (1/(n*PI)) * cos(2*n*PI*x/L) * sin(n*PI / DIM);
    return s;
}


/**
 * Given numbers, the function generates the next set of probabilities
 * based on these numbers.
 *
 * @param int           the numbers to generate the probabilities from
 * @param int           the length of the numbers array (should be the same
                        for the probabilities array);
 * @param float         array of probabilities to write to
*/
void next_probabilities(int *numbers, int N, float *probabilities) {
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

/**
 * Create a new particle
 * 
 * @param particle      the particle to create
 * @param int           the number of energy levels
 * @param float         array of the probability of every energy level
 * @param float         the mass of the particle
*/
void create_particle(particle *p, int N, float *probabilities, float mass) {
    p->mass = mass;
    p->waveCount = N;
    p->wave = (_wave_function **) malloc(N*sizeof(_wave_function));
    for(int i=0;i<N;i++) {
        p->wave[i] = (_wave_function *) malloc(sizeof(_wave_function));
        _create_wave(p->wave[i], i+1, mass, probabilities[i]);
    }
}

/**
 * Get the probability of finding a particle in two dimensions
 *
 * @param particle      the particle to find the probability of
 * @param float         the x-position in which to find the probability at
 * @param float         the y-position in which to find the probability at
 * @return              the probability of finding the particle at the point (x, y)
*/
float probability(particle *p, float x, float y) {
    if(x > 0 && x < L && y > 0 && y < L) {
        float probability = 0;
        for(int i=0;i<p->waveCount;i++)
            probability += (1.0/(L*L) * _probability_1d(p->wave[i], x) *
                           _probability_1d(p->wave[i], y)) * p->wave[i]->probability;
        return probability;
    }
    return 0;
}



int main(int argc, const char *argv[]) { 

    particle p;
    int N = 10;
    float *pro = (float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++) pro[i] = 0.1f;
    create_particle(&p, N, pro, 2.5);

    printf("Testing on a particle\n");
    printf("The particle has mass %f and has %d energy levels\n",
           p.mass, p.waveCount);
    for(int i=0;i<p.waveCount;i++)
        printf("\tWave %d has %f energy and %.3f probability\n", i+1, p.wave[i]->energy,
               p.wave[i]->probability);

    float all_proba = 0;
    float max_proba = 0, max_x, max_y;
    for(float x=0;x<=L;x+=E) {
        for(float y=0;y<=L;y+=E) {
            float pn = probability(&p, x, y);
            if(pn > max_proba) {
                max_proba = pn;
                max_x = x;
                max_y = y;
            }
            all_proba += probability(&p, x, y);
        }
    }
    printf("The probability inside the box is %f\n", all_proba);

    float x = 0.5,
          y = 0.5;
    printf("The maximum probability is %f%% at (%.3f, %.3f)\n", max_proba*100, max_x, max_y);
    printf("The probability of finding the particle in (%f, %f) is %.5f%% or %.3f%% of the maximum\n",
           x, y, probability(&p, x, y)*100, probability(&p, x, y)*100/max_proba);

    printf("Testing on generating new probabilities\n");
    int *numbers = (int*) malloc(N*sizeof(int));
    float *proba = (float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++) numbers[i] = 0;
    numbers[0] = 1;
    for(int i=0;i<10;i++) {
        next_probabilities(numbers, N, proba);
        for(int i=0;i<N;i++)
            printf("%.3f\t", proba[i]);
        printf("\n");
    }
    return 0;
}
