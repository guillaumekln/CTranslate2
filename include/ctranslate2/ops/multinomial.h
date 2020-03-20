#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Multinomial : public UnaryOp {
    public:
      Multinomial(const dim_t sample_size = 1, const bool replacement = false);
      void operator()(const StorageView& input, StorageView& output) const override;

    private:
      const dim_t _sample_size;
      const bool _replacement;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
