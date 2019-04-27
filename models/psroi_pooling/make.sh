#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

cd src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
${CUDA_PATH}/bin/nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../../
python build.py
