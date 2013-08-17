NCC=nvcc
CFLAGS=-arch=sm_20 -use_fast_math

all:
	$(NCC) $(CFLAGS) pbox.cu -o pbox
clean:
	rm pbox
run:
	./pbox
