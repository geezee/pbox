NCC=nvcc
CFLAGS=-arch=sm_20 -use_fast_math
OPENGLFLAGS=-lglut -lGL

all:
	$(NCC) $(CFLAGS) $(OPENGLFLAGS) pbox.cu -o pbox
clean:
	rm pbox
run:
	./pbox 10
