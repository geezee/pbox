NCC=nvcc
CFLAGS=-arch=sm_20 -use_fast_math -lcudart
OPENGLFLAGS=-lglut -lGL -lGLU
GCCFLAGS=-lcurses
FILES=pbox.cu
BINARY_NAME=pbox

all:
	$(NCC) $(CFLAGS) $(OPENGLFLAGS) $(GCCFLAGS) $(FILES) -o $(BINARY_NAME)
clean:
	rm pbox
run:
	./pbox
