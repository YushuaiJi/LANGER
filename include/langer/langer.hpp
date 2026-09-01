#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace langer {

using VectorId = std::uint32_t;
using CentroidId = std::uint32_t;

struct Predicate {
  std::size_t attribute = 0;
  std::int32_t value = 0;

  bool operator==(const Predicate& other) const noexcept {
    return attribute == other.attribute && value == other.value;
  }
};

struct PredicateHash {
  std::size_t operator()(const Predicate& predicate) const noexcept;
};

struct BuildParameters {
  // Zero selects max(1, n / 1000), the paper's default scale.
  std::size_t num_clusters = 0;
  std::size_t clustering_iterations = 20;
  float selectivity_cutoff = 0.01F;
  std::size_t max_degree = 20;
  std::size_t construction_ef = 200;
  std::size_t entry_point_count = 8;
  std::uint64_t random_seed = 42;
};

struct SearchParameters {
  std::size_t top_k = 10;
  std::size_t ef_search = 100;
  std::size_t cluster_scan_budget = 32;
};

struct SearchResult {
  VectorId id = 0;
  float distance = 0.0F;
};

class Index {
 public:
  static Index build(const std::vector<float>& vectors,
                     std::size_t num_vectors,
                     std::size_t dimension,
                     const std::vector<std::int32_t>& labels,
                     std::size_t num_attributes,
                     const BuildParameters& parameters = {});

  std::vector<SearchResult> search(
      const std::vector<float>& query,
      const std::vector<Predicate>& predicates,
      const SearchParameters& parameters = {}) const;

  void save(const std::string& path) const;
  static Index load(const std::string& path);

  std::size_t size() const noexcept { return num_vectors_; }
  std::size_t dimension() const noexcept { return dimension_; }
  std::size_t num_attributes() const noexcept { return num_attributes_; }
  std::size_t num_clusters() const noexcept { return cluster_sizes_.size(); }
  std::size_t num_value_graphs() const noexcept { return graphs_.size(); }
  std::size_t max_observed_out_degree() const noexcept;

 private:
  using PredicateCounts =
      std::unordered_map<Predicate, std::uint32_t, PredicateHash>;

  struct ValueGraph {
    std::vector<std::vector<CentroidId>> adjacency;
    std::vector<CentroidId> entry_centroids;
  };

  std::size_t num_vectors_ = 0;
  std::size_t dimension_ = 0;
  std::size_t num_attributes_ = 0;
  BuildParameters build_parameters_;
  std::vector<float> vectors_;
  std::vector<std::int32_t> labels_;
  std::vector<float> feature_mean_;
  std::vector<float> feature_stddev_;
  std::vector<float> centroids_;
  std::vector<std::vector<VectorId>> posting_lists_;
  std::vector<std::uint32_t> cluster_sizes_;
  std::vector<PredicateCounts> cluster_predicate_counts_;
  std::unordered_map<Predicate, ValueGraph, PredicateHash> graphs_;
};

}  // namespace langer
