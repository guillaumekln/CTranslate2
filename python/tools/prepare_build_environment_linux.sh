#! /bin/bash

set -e
set -x

# Install CUDA 10.0, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.0/centos7-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.0/centos7-x86_64/devel/Dockerfile
CUDA_VERSION=10.0.130
CUDA_PKG_VERSION=10-0-$CUDA_VERSION-1
curl -o cuda-repo.rpm https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-$CUDA_VERSION-1.x86_64.rpm
rpm --install cuda-repo.rpm
yum clean expire-cache
yum install --setopt=obsoletes=0 -y cuda-nvcc-$CUDA_PKG_VERSION cuda-cudart-dev-$CUDA_PKG_VERSION cuda-cublas-dev-$CUDA_PKG_VERSION
ln -s cuda-10.0 /usr/local/cuda

# Maximum GCC version supported by CUDA 10.0 is GCC 7.
yum install -y devtoolset-7
source /opt/rh/devtoolset-7/enable

ONEAPI_VERSION=2021.1.1
MKL_BUILD=52
DNNL_BUILD=55
yum install -y yum-utils
yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD intel-oneapi-dnnl-devel-$ONEAPI_VERSION-$DNNL_BUILD
echo "/opt/intel/oneapi/dnnl/latest/cpu_gomp/lib" > /etc/ld.so.conf.d/intel-dnnl.conf
ldconfig

pip install "cmake==3.18.4"

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DLIB_ONLY=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DWITH_CUDA=ON -DCUDA_NVCC_FLAGS="-Xfatbin -compress-all" -DCUDA_ARCH_LIST="Common" ..
make -j$(nproc) install
cd ..
rm -r build-release

/usr/bin/cp --remove-destination README.md python/README.md
