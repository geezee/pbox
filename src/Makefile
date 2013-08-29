NCC=nvcc
CFLAGS=-arch=sm_20 -use_fast_math -lcudart
OPENGLFLAGS=-lglut -lGL -lGLU
GCCFLAGS=-lcurses

all:
	$(NCC) $(CFLAGS) $(OPENGLFLAGS) $(GCCFLAGS) pbox.cu -o pbox
clean:
	rm pbox
run:
	./pbox
