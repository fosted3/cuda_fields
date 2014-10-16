CUDA_CC=/opt/cuda/bin/nvcc
C_CC=gcc
CPP_CC=g++
INCLUDES=/opt/cuda/samples/common/inc
COMPUTE=arch=compute_30,code=sm_30
CFLAGS=-c -Wall -g -O3 -std=c++11 -Wextra -march=native -mtune=native
LDFLAGS=-lfreeimage
CPP_BIN=bin/image_write
CUDA_BIN=bin/cuda_fields
DIRS=bin/ build/

ALL: $(CUDA_BIN) $(CPP_BIN)

$(CUDA_BIN): build/cuda_fields.o
	$(CUDA_CC) -ccbin $(C_CC) -m64 -gencode $(COMPUTE) -o $(CUDA_BIN) build/cuda_fields.o

build/cuda_fields.o: src/cuda_fields.cu
	mkdir -p $(DIRS)
	$(CUDA_CC) -ccbin $(C_CC) -I$(INCLUDES) -m64 -gencode $(COMPUTE) -o build/cuda_fields.o -c src/cuda_fields.cu

$(CPP_BIN): build/image_write.o
	$(CPP_CC) build/image_write.o -o $(CPP_BIN) $(LDFLAGS)

build/image_write.o: src/image_write.cpp
	mkdir -p $(DIRS)
	$(CPP_CC) $(CFLAGS) src/image_write.cpp -o build/image_write.o

clean:
	rm -f build/*.o $(CUDA_BIN) $(CPP_BIN)
