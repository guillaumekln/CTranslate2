#include "ctranslate2/vocabulary_map.h"

#include <fstream>

namespace ctranslate2 {

  VocabularyMap::VocabularyMap(const std::string& map_path, const Vocabulary& vocabulary) {
    std::ifstream map_file(map_path);
    if (!map_file.is_open())
      return;

    std::string line;
    while (std::getline(map_file, line)) {
      std::string token;
      std::string key;
      std::vector<size_t> values;
      bool target = false;
      size_t ngram = 1;

      for (size_t i = 0; i < line.length(); ++i) {
        if (line[i] == '\t') {
          target = true;
          std::swap(key, token);
        } else if (line[i] == ' ') {
          if (target) {
            values.push_back(vocabulary.to_id(token));
            token.clear();
          } else {
            token += line[i];
            ++ngram;
          }
        } else
          token += line[i];
      }

      if (!token.empty())
        values.push_back(vocabulary.to_id(token));

      if (ngram > _map_rules.size())
        _map_rules.resize(ngram);

      _map_rules[ngram - 1][key] = values;
    }

    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::unk_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::bos_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::eos_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::pad_token));

    // The field marked by the empty string are common tokens that are always candidates.
    auto it = _map_rules[0].find("");
    if (it != _map_rules[0].end())
      _fixed_candidates.insert(it->second.begin(), it->second.end());
  }

  bool VocabularyMap::empty() const {
    return _map_rules.empty();
  }

}
