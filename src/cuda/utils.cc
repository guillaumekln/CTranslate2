#include "./utils.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ctranslate2/primitives/primitives.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

    std::string cublasGetStatusString(cublasStatus_t status)
    {
      switch (status)
      {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "UNKNOWN";
      }
    }

    // See https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html.
    struct CachingAllocatorConfig {
      unsigned int bin_growth = 4;
      unsigned int min_bin = 3;
      unsigned int max_bin = 12;
      size_t max_cached_bytes = 200 * (1 << 20);  // 200MB
    };

    static CachingAllocatorConfig get_caching_allocator_config() {
      CachingAllocatorConfig config;
      const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
      if (config_env) {
        const std::vector<std::string> values = split_string(config_env, ',');
        if (values.size() != 4)
          throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                      "should have format: "
                                      "bin_growth,min_bin,max_bin,max_cached_bytes");
        config.bin_growth = std::stoul(values[0]);
        config.min_bin = std::stoul(values[1]);
        config.max_bin = std::stoul(values[2]);
        config.max_cached_bytes = std::stoull(values[3]);
      }
      return config;
    }

    class CudaContext {
    public:
      CudaContext() {
        CUDA_CHECK(cudaGetDevice(&_device));
        //CUDA_CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&_handle));
        //CUBLAS_CHECK(cublasSetStream(_handle, _stream));

        const auto allocator_config = get_caching_allocator_config();
        _allocator.reset(new cub::CachingDeviceAllocator(allocator_config.bin_growth,
                                                         allocator_config.min_bin,
                                                         allocator_config.max_bin,
                                                         allocator_config.max_cached_bytes));
      }

      ~CudaContext() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        _allocator.reset();
        cublasDestroy(_handle);
        //cudaStreamDestroy(_stream);
      }

      cublasHandle_t get_cublas_handle() const {
        return _handle;
      }

      cudaStream_t get_stream() const {
        return 0;
        //return _stream;
      }

      cub::CachingDeviceAllocator& get_allocator() {
        return *_allocator;
      }

    private:
      int _device;
      //cudaStream_t _stream;
      cublasHandle_t _handle;
      std::unique_ptr<cub::CachingDeviceAllocator> _allocator;
    };

    static CudaContext& get_context() {
      // We create a separate CUDA context for each host thread. The context is destroyed
      // when the thread exits.
      static thread_local CudaContext context;
      return context;
    }

    cudaStream_t get_cuda_stream() {
      return get_context().get_stream();
    }

    cublasHandle_t get_cublas_handle() {
      return get_context().get_cublas_handle();
    }

    cub::CachingDeviceAllocator& get_allocator() {
      return get_context().get_allocator();
    }

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    const cudaDeviceProp& get_device_properties(int device) {
      static thread_local std::vector<std::unique_ptr<cudaDeviceProp>> cache;

      if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
      }
      if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
      }

      auto& device_prop = cache[device];
      if (!device_prop) {
        device_prop.reset(new cudaDeviceProp());
        CUDA_CHECK(cudaGetDeviceProperties(device_prop.get(), device));
      }
      return *device_prop;
    }

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool gpu_supports_int8(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool gpu_has_int8_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 7 || (device_prop.major == 7 && device_prop.minor >= 2);
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }

    ThrustAllocator::value_type* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
      return reinterpret_cast<ThrustAllocator::value_type*>(
        primitives<Device::CUDA>::alloc_data(num_bytes));
    }

    void ThrustAllocator::deallocate(ThrustAllocator::value_type* p, size_t) {
      return primitives<Device::CUDA>::free_data(p);
    }

    ThrustAllocator& get_thrust_allocator() {
      static ThrustAllocator thrust_allocator;
      return thrust_allocator;
    }

  }
}
