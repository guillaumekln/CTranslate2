#pragma once

// Low-level (BLAS-like) primitives.

#include "cpu_generic.h"

#ifdef WITH_MKL
#  include "cpu_mkl.h"
#endif

#ifdef WITH_CUDA
#  include "gpu_cuda.h"
#endif
