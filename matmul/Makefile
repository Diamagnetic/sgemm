CUDA_PATH ?= /usr/local/cuda
CUDA_RUNTIME := $(CUDA_PATH)/include/cuda_runtime.h

NVXX := $(CUDA_PATH)/bin/nvcc
GXX := g++

GXXDEBUGFLAGS := -g -O0
NVXXFLAGS := -G -O0

LDFLAGS := -L $(CUDA_PATH)/lib64 -lcudart

CUHFILES := matmul.cuh

ALL_HFILES := $(CUHFILES)

PRODUCT = matmul

all: $(PRODUCT)

matmul_cu.o: matmul.cu $(CUHFILES)
	$(NVXX) $(NVXXFLAGS) -c $< -o $@

matmul.o: matmul.cpp $(ALL_HFILES)
	$(GXX) $(GXXFLAGS) $(GXXDEBUGFLAGS) -c $< -o $@

matmul: matmul.o matmul_cu.o
	$(GXX) $^ -o $@ $(LDFLAGS)

clean:
	rm *.o $(PRODUCT)
