#include <langer/langer.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

int main() {
  constexpr std::size_t n = 180;
  constexpr std::size_t dimension = 6;
  constexpr std::size_t attributes = 2;
  std::vector<float> vectors(n * dimension);
  std::vector<std::int32_t> labels(n * attributes);

  for (std::size_t i = 0; i < n; ++i) {
    labels[i * attributes] = static_cast<std::int32_t>(i % 5);
    labels[i * attributes + 1] = static_cast<std::int32_t>((i / 5) % 3);
    for (std::size_t j = 0; j < dimension; ++j) {
      vectors[i * dimension + j] =
          static_cast<float>(labels[i * attributes]) +
          0.02F * static_cast<float>(i) +
          0.01F * std::cos(static_cast<float>((i + 2) * (j + 1)));
    }
  }

  langer::BuildParameters build;
  build.num_clusters = 12;
  build.clustering_iterations = 6;
  build.max_degree = 5;
  build.construction_ef = 10;
  auto index = langer::Index::build(
      vectors, n, dimension, labels, attributes, build);

  assert(index.num_clusters() == build.num_clusters);
  assert(index.max_observed_out_degree() <= build.max_degree);

  const std::size_t query_id = 41;
  std::vector<float> query(
      vectors.begin() + static_cast<std::ptrdiff_t>(query_id * dimension),
      vectors.begin() + static_cast<std::ptrdiff_t>((query_id + 1) * dimension));
  const std::vector<langer::Predicate> predicates = {
      {0, labels[query_id * attributes]},
      {1, labels[query_id * attributes + 1]}};

  langer::SearchParameters search;
  search.top_k = 10;
  search.ef_search = index.num_clusters();
  search.cluster_scan_budget = index.num_clusters();
  const auto result = index.search(query, predicates, search);
  assert(!result.empty());
  assert(result.front().id == query_id);
  for (const auto& hit : result) {
    assert(labels[hit.id * attributes] == predicates[0].value);
    assert(labels[hit.id * attributes + 1] == predicates[1].value);
  }

  const std::string path = "langer_test_index.langer";
  index.save(path);
  auto loaded = langer::Index::load(path);
  const auto loaded_result = loaded.search(query, predicates, search);
  assert(!loaded_result.empty());
  assert(loaded_result.front().id == query_id);
  std::remove(path.c_str());
  return 0;
}
