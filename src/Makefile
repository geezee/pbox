NCC=nvcc
CFLAGS=-arch=sm_20 -use_fast_math -lcudart
OPENGLFLAGS=-lglut -lGL -lGLU

all:
	$(NCC) $(CFLAGS) $(OPENGLFLAGS) pbox.cu -o pbox
clean:
	rm pbox
run:
	./pbox
