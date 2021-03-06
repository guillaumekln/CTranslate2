#pragma once

#include <unordered_map>
#include <memory>

#include "ctranslate2/vocabulary.h"
#include "ctranslate2/vocabulary_map.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/layers/decoder.h"

namespace ctranslate2 {
  namespace models {

    static const size_t current_binary_version = 2;

    // Base class for models.
    class Model {
    public:
      static std::shared_ptr<Model> load(const std::string& path,
                                         const std::string& device,
                                         int device_index = 0,
                                         const std::string& computeType = "default");
      static std::shared_ptr<Model> load(const std::string& path,
                                         Device device,
                                         int device_index = 0,
                                         ComputeType computeType = ComputeType::DEFAULT);

      virtual ~Model() = default;
      virtual size_t current_spec_revision() const;

      Device device() const;
      ScopedDeviceSetter get_scoped_device_setter() const;

      const Vocabulary& get_source_vocabulary() const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap& get_vocabulary_map() const;

      const StorageView* get_variable_if_exists(const std::string& name) const;
      const StorageView& get_variable(const std::string& name) const;
      const std::unordered_map<std::string, StorageView>& get_variables() const;

      // Makes new graph to execute this model. Graphs returned by these function
      // should support being executed in parallel without duplicating the model
      // data (i.e. the weights).
      virtual std::unique_ptr<layers::Encoder> make_encoder() const = 0;
      virtual std::unique_ptr<layers::Decoder> make_decoder() const = 0;

    protected:
      Model(const std::string& path, size_t spec_revision);

      // Models can override these methods to execute some transformations if needed
      // (e.g. a variable name changed in a newer spec revision).
      virtual void register_variable(const std::string& name, StorageView& variable);
      virtual void finalize();
      StorageView* get_scale(const std::string& scale_name, DataType dataType);

      void set_device(Device type, int index = 0);
      void set_computType(ComputeType type);

      Device _device;
      int _device_index;
      const Vocabulary _source_vocabulary;
      const Vocabulary _target_vocabulary;
      const VocabularyMap _vocabulary_map;
      std::unordered_map<std::string, StorageView> _variable_index;
      size_t _spec_revision;
      ComputeType _computeType = ComputeType::DEFAULT;

      void convert_data_if_need(bool support_int8,
                                bool support_int16,
                                const std::string& name,
                                StorageView& variable,
                                std::vector<std::pair<std::string, StorageView>>& variables_to_add,
                                std::vector<std::string>& variables_to_remove);
    };

  }
}
