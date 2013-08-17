#include <stdio.h>
#include <cuda.h>
#include <math.h>


#define PI   3.1415
#define L    1.0          // the length of the box
#define DIM  500          // the dimensions of the window
#define E    (L/DIM)
#define HBAR 1.054        // without the e-34

#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b


typedef struct _wave_function {
    int n;          // the energy level of the wave function
    float energy;   // the energy of the particle at that level
} _wave_function;

typedef struct particle {
    float mass;           // the mass of the particle
    _wave_function *wave; // the wave function of the particle 
} particle;


/**
  *Create a new particle
 *
  *@param particle      the particle to create
  *@param int           the energy level of the particle
  *@param float         the mass of the particle
*/
void create_particle(particle *p, int n, float mass) {
    p->mass = mass;
    p->wave = (_wave_function *) malloc(sizeof(_wave_function));
    p->wave->n = n;
    p->wave->energy = HBAR*HBAR * PI*PI * n*n / (mass * L*L);
}

/**
  *Get the probability of finding a particle in one dimension
 *
  *@param particle      the particle to find the probability of
  *@param float         the position in which to find the probability at
  *@return              the probability of finding the particle at the point x
*/
float _probability_1d(particle *p, float x) {
    int n   =  p->wave->n;
    float s =  E/L - (1/(n*PI)) * cos(2*n*PI*x/L) * sin(n*PI / DIM);
    return s;
}

/**
  *Get the probability of finding a particle in two dimensions
 *
  *@param particle      the particle to find the probability of
  *@param float         the x-position in which to find the probability at
  *@param float         the y-position in which to find the probability at
  *@return              the probability of finding the particle at the point (x, y)
*/
float probability(particle *p, float x, float y) {
    if(x > 0 && x < L && y > 0 && y < L) {
        return 1.0/(L*L) * _probability_1d(p, x) * _probability_1d(p, y);
    }
    return 0;
}

int main(int argc, const char *argv[]) { 

    particle p;
    create_particle(&p, 3, 2.5);

    printf("The probability has mass %f\n\tIs in the %dth energy level\n\tHas %f of energy\n",
           p.mass, p.wave->n, p.wave->energy);

    float all_proba = 0;
    for(float x=0;x<=L;x+=E)
        for(float y=0;y<=L;y+=E)
            all_proba += probability(&p, x, y);
    printf("The probability inside the box is %f\n", all_proba);

    float x = 1.0*rand()/RAND_MAX;
    float y = 1.0*rand()/RAND_MAX;
    printf("The probability of finding the particle in (%f, %f) is %f%%\n",
           x, y, probability(&p, x, y)*100);

    return 0;
}
