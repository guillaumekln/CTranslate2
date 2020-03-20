#include "ctranslate2/ops/multinomial.h"

#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <typename In, typename Out>
    static void multinomial_kernel(const In* input,
                                   const dim_t batch_size,
                                   const dim_t class_size,
                                   const dim_t sample_size,
                                   const bool replacement,
                                   Out* output) {
      auto& generator = get_random_generator();
      auto* cum_dist = static_cast<In*>(primitives<>::alloc_data(class_size * sizeof (In)));
      std::uniform_real_distribution<double> uniform;

      for (dim_t i = 0; i < batch_size; ++i) {
        const In* in = input + i * class_size;

        // Compute the normalized cumulative distribution.
        const In sum = *std::prev(std::partial_sum(in, in + class_size, cum_dist));
        primitives<>::mul(In(1) / sum, cum_dist, class_size);

        Out* out = output + i * sample_size;
        for (dim_t j = 0; j < sample_size; ++j) {
          // Draw an index such that cum_dist[index-1] < uniform_sample < cum_dist[index].
          const double uniform_sample = uniform(generator);
          const Out index = std::distance(
            cum_dist,
            std::lower_bound(cum_dist, cum_dist + class_size, static_cast<In>(uniform_sample)));

          out[j] = index;

          // Prevent index from being sampled again when sampling without replacement.
          if (!replacement && j + 1 < sample_size) {
            // Remove contribution of the selected index in the cumulative distribution.
            const In mass = cum_dist[index] - (index > 0 ? cum_dist[index - 1] : 0);
            primitives<>::sub(mass, cum_dist + index, class_size - index);

            // Renormalize by the new sum.
            const In new_sum = In(1) - mass;
            primitives<>::mul(In(1) / new_sum, cum_dist, class_size);
          }
        }

      }

      primitives<>::free_data(cum_dist);
    }

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      const dim_t class_size = input.dim(-1);
      const dim_t batch_size = input.size() / class_size;
      multinomial_kernel(input.data<T>(),
                         batch_size,
                         class_size,
                         _sample_size,
                         _replacement,
                         output.data<int32_t>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CPU, T>(const StorageView& input,      \
                                         StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
