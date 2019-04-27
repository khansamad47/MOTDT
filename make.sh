#!/usr/bin/env bash

echo "build solvers..."
python bbox_setup.py build_ext --inplace

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61"
cd utils/nms/
echo "Compiling nms.."

nvcc -c -o nms_kernel.cu.o nms_kernel.cu --compiler-options -fPIC $CUDA_ARCH
cd ../../
echo "build psroi_pooling..."
cd models/psroi_pooling
sh make.sh
