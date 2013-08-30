#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curses.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define PI   3.141592653589793
#define L    1.0            // the length of the box
#define DIM  512            // the dimensions of the window (should be 2^n)
#define E    (L/DIM)        // epsilon
#define HBAR 1.054          // without the e-34


#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b


/* The struct that describes a particle */
typedef struct particle {
    float mass;            // mass of the particle
    int energy_levels;     // the number of energy levels
    float *probabilities;  // the probabilitis of every energy level
} particle;


/** global variables used in the code */
cudaGraphicsResource *resource;
GLuint   buffer;
float    INCREASE_TIME = 0.01;           // number to increase the time by
particle p;                              // the particle
float    t             = INCREASE_TIME;  // starting time
int      frames        = 0;              // number of frames
float    total_time    = 0.0f;           // total time
int      PAUSE         = 0;              // whether the animation is paused

/** ncurses variables */
WINDOW* window;
int ncols, nrows;


/**
 * Function headers
*/
float max(float*, int);
void  next_probabilities(float, int, float*);
void  create_particle(particle*, int, float);
float probability(particle*, float, float);
float max_probability(particle*);
void  initGL(int*, char**);
void  display();
void  key(unsigned char, int, int);
void  free_resources();
void  createVBO(GLuint*, cudaGraphicsResource**, unsigned int);
void  runCuda(cudaGraphicsResource**);
void  launch_kernel(uchar4);
void  runCuda(cudaGraphicsResource **resource);
void  run(int, char**);
void  usage(char*);
/** For ncurses */
void  clear_row(int);
void  cmd_display();
void  init_curses();


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
    free(numbers);
    return m;
}

/**
 * The set of probabilities can be expressed as a bunch of sine waves
 * For more information read section 4.4 (Next set of probabilities) on
 * page 9 in the paper
 *
 * @param float               the time
 * @param int                 the length of the array
 * @param float*              where to store the next set of probabilities
*/
void next_probabilities(float t, int N, float* probabilities) {
    float sum = 0;
    for(int i=0;i<N;i++) {
        probabilities[i] = abs(sin(pow(10, 1-i)*t));
        sum += probabilities[i];
    }
    // Normalize the probabilities
    for(int i=0;i<N;i++) probabilities[i] /= sum;
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
    float *proba;
    proba = &probability[0];
    free(probability);
    return *proba;
}

/**
 * A function that finds the maximum probability in the space of finding the particle
 * This function is used to map the probabilities [0, max] |-> [0, 255]
 * 
 * @param particle*                  the particle
 * @return                           the highest probability
*/
float max_probability(particle *p) {
    float *devProbabilities, *devMap;
    float *map;

    map = (float*) malloc(DIM * DIM * sizeof(float));
    cudaMalloc((void**) &devProbabilities, p->energy_levels * sizeof(float));
    cudaMalloc((void**) &devMap, DIM * DIM * sizeof(float));
    
    cudaMemcpy(devProbabilities, p->probabilities, p->energy_levels * sizeof(float), cudaMemcpyHostToDevice);
    cuda_probability_to_map<<<32, 256>>>(devProbabilities, p->energy_levels, devMap);
    cudaMemcpy(map, devMap, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devProbabilities);
    cudaFree(devMap);

    return max(map, DIM*DIM);

    /*
    // This algorithm failed for many waves
    // I had to fall back to the original bruteforce algorithm written above
    float probability = E/L, p_1d = 0.0;
    int n;
    for(int i=0;i<10;i++) {
        n = i+1;
        p_1d = E/L + 1.0/(n*PI) * sin(n*PI*E);
        probability += p->probabilities[i] * p_1d * p_1d;
    }

    return probability;
    */
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

__global__
void kernel(uchar4 *ptr, float *probabilities, int N, float max_proba) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while(i < DIM*DIM) {
        float x = L*((int) (i%DIM))/DIM;
        float y = L*((int) (i/DIM))/DIM;
        float p = cuda_probability_2d_device(probabilities, N, x, y)/max_proba;
        float e = 0;
        for(int j=0;j<N;j++) e += probabilities[j]*energy(1.0, j+1);
        e /= highest_energy(1.0, N);

        ptr[i].x = 255*p*e;
        ptr[i].y = 20*p;
        ptr[i].z = 255*p*(1-e);

        i += offset;
    }
}


/////////////////////////// GUI PART /////////////////////////////////////
/**
 * Initialize the OpenGL environment
 *
 * @param int       length of next paramater
 * @param char      the parameters to the environment
*/
void initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow("Particle in a box simulation");
    glutDisplayFunc(display);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
}

/**
 * Manage key strokes
 *
 * @param unsigned char         the character pressed
 * @param int                   x-location of the pressing
 * @param int                   y-location of the pressing
*/
void key(unsigned char k, int x, int y) {
    switch(k) {
        case 27:
            free_resources();
            endwin();
            exit(0);
        case '.':
            INCREASE_TIME *= 1.05; break;
        case ',':
            INCREASE_TIME = MAX(0.01, INCREASE_TIME*0.95); break;
        case '0':
            t = 0.01f; break;
        case ' ':
            PAUSE = 1-PAUSE;
            if(PAUSE == 1) {
                glutPostRedisplay();
            } break;
        case 'm':
            t += INCREASE_TIME; glutPostRedisplay(); break;
        case 'n':
            t -= INCREASE_TIME; glutPostRedisplay(); break;
    }
}

/**
 * Free the OpenGL resources
*/
void free_resources() {
    cudaGraphicsUnregisterResource(resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &buffer);
}

/**
 * Function that creates the buffer and the resource for the environment
 *
 * @param GLuint                    the buffer used by OpenGL
 * @param cudaGraphicsResource      the cuda resource to link to the buffer
*/
void createVBO(GLuint *buffer, cudaGraphicsResource **resource,
               unsigned int flags) {
    glGenBuffers(1, buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *buffer);

    unsigned int size = DIM * DIM * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(resource, *buffer, flags);
}

/**
 * Kernel launcher
 *
 * @param uchar4                the pixel array
*/
void launch_kernel(uchar4 *pos) {
    float *devProbabilities;
    int N = p.energy_levels;
    cudaMalloc((void**) &devProbabilities, N*sizeof(float));
    cudaMemcpy(devProbabilities, p.probabilities, N*sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<32, 256>>>(pos, devProbabilities, N, max_probability(&p));
    next_probabilities(t, N, (p.probabilities));
    cudaFree(devProbabilities);
}

/**
 * Function that creates the resources for the kernel and launches it
 *
 * @param cudaGraphicsResource  the cuda resource
*/
void runCuda(cudaGraphicsResource **resource) {
    uchar4 *devPtr;
    size_t size;

    cudaGraphicsMapResources(1, resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**) &devPtr, &size, *resource);
    launch_kernel(devPtr);
    cudaGraphicsUnmapResources(1, resource, 0);
}


/**
 * The function that runs everything
 *
 * @param int       length of next paramater
 * @param char      the parameters to the environment
*/
void run(int argc, char **argv) {
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);
    cudaGLSetGLDevice(dev);

    initGL(&argc, argv);

    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    createVBO(&buffer, &resource, cudaGraphicsMapFlagsWriteDiscard);
    runCuda(&resource);

    glutMainLoop();
}


/**
 * Prints out the help message of the program
 *
 * @param char      the name of the program
*/
void usage(char* program_name) {
    int y = 7+p.energy_levels/2+2;
    attron(COLOR_PAIR(2));
    move(y, 0);
    clear_row(y);
    printw("Keyboard Shortcuts");
    attroff(COLOR_PAIR(2));
    move(y+1, 0);
    attron(COLOR_PAIR(5));
    printw(".\t\tIncrease the time offset\n");
    printw(",\t\tDescrease the time offset\n");
    printw("n\t\tGo back one frame\n");
    printw("m\t\tGo forward one framed\n");
    printw("<space>\t\tToggle pausing\n");
    printw("0\t\tReset the animation\n");
    printw("<esc>\t\tQuit\n\n");
    attroff(COLOR_PAIR(5));
    attron(COLOR_PAIR(2));
    clear_row(y+9);
    printw("Usage");
    attroff(COLOR_PAIR(2));
    move(y+10, 0);
    attron(COLOR_PAIR(5));
    printw("%s [n]\nDESCRIPTION\n  n\n    The number of wave functions to simulate", program_name);
    attroff(COLOR_PAIR(5));
}

/**
 * Function that clears the rows
 */
void clear_row(int y) {
    move(y, 0);
    for(int i=0;i<ncols;i++) {
        move(y, i);
        delch();
        insch(' ');
    }
    move(y, 0);
}

/**
 * Create the command line interface
*/
void cmd_display() {
    // Draw the title
    attron(COLOR_PAIR(2));
    clear_row(0);
    printw("Simulating a two dimensional box in a square quantum box");

    // Draw the stats about the program
    attron(COLOR_PAIR(4));
    move(2, 0); printw("Simulating with %d wave functions", p.energy_levels);
    attroff(COLOR_PAIR(4));
    attron(COLOR_PAIR(2));
    clear_row(3);
    move(3, 0); printw("Time");
    move(3, 1*ncols/4); printw("Frame");
    move(3, 2*ncols/4); printw("Time per frame");
    move(3, 3*ncols/4); printw("Increment");
    attroff(COLOR_PAIR(2));
    attron(COLOR_PAIR(5));
    move(4, 0); printw("%.2f", t);
    move(4, 1*ncols/4); printw("%d", frames);
    move(4, 2*ncols/4); printw("%.2fms", total_time/frames);
    move(4, 3*ncols/4); printw("%.5f", INCREASE_TIME);
    attroff(COLOR_PAIR(5));

    // Write out the probabilities
    attron(COLOR_PAIR(2));
    move(6, 0);
    clear_row(6);
    for(int i=0;i<4;i++) {
        move(6, i*ncols/4);
        printw(i%2 == 0 ? "Wave number" : "Probability");
    }
    attroff(COLOR_PAIR(2));
    move(6, ncols/2-ncols/16);
    delch();
    insch(' ');

    attron(COLOR_PAIR(5));
    for(int i=0;i<p.energy_levels;i++) {
        move(7+i/2, i%2 * ncols/2);
        printw("%d", i+1);
        move(7+i/2, i%2 * ncols/2 + ncols/4);
        printw("%.2f%%", 100*p.probabilities[i]);
    }
    attroff(COLOR_PAIR(5));

    move(0, ncols);
    refresh();
}

/**
 * The display function that runs on every iteration
*/
void display() {
    cudaEvent_t start, stop;
    cudaEventCreate(&stop);
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4, NULL, GL_DYNAMIC_DRAW_ARB);
    cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone);
    glClear(GL_COLOR_BUFFER_BIT);
    runCuda(&resource);
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glutSwapBuffers();

    t += (1-PAUSE)*INCREASE_TIME;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float diff;
    cudaEventElapsedTime(&diff, start, stop);

    glutPostRedisplay();

    if(!PAUSE) {
        total_time += diff;
        frames++;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cmd_display();
}

/**
 * Function that initializes the curses library
 * Sets up colors and fills the background
*/
void init_curses() {
    /** Initialize the ncurses variables */
    window = initscr();
    start_color();
    cbreak();
    noecho();
    getmaxyx(window, nrows, ncols);

    /** Initialize the colors that will be used in ncurses */
    init_pair(1, COLOR_BLACK, COLOR_BLACK); // background
    init_pair(2, COLOR_WHITE, COLOR_BLUE);  // titles
    init_pair(3, COLOR_RED, COLOR_BLACK);   // errors
    init_pair(4, COLOR_GREEN, COLOR_BLACK); // messages
    init_pair(5, COLOR_WHITE, COLOR_BLACK); // text

    clear();
    // Fill the background
    attron(COLOR_PAIR(1));
    for(int x=0;x<nrows;x++) {
        for(int y=0;y<ncols;y++) {
            move(x, y);
            delch();
            insch(' ');
        }
    }
    attroff(COLOR_PAIR(1));
    refresh();
}

///////////////////////////// MAIN FUNCTION //////////////////////////////////
int main(int argc, char *argv[]) { 
    int N = 5;

    if(argc > 1) {
        if(atoi(argv[1]) > 0) {
            N = atoi(argv[1]);
        } else {
            printf("\033[01;31mWARNING: You are trying to simulate a negative ");
            printf("number of wave functions.\nWill fall back to %d (default)\033[22;m\n",N);
        }
    }

    create_particle(&p, N, 0.003f);
    init_curses();
    usage(argv[0]);

    run(argc, argv);
    return 0;
}
