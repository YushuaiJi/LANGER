#include <langer/langer.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t n = 320;
  constexpr std::size_t dimension = 8;
  constexpr std::size_t attributes = 3;

  std::vector<float> vectors(n * dimension);
  std::vector<std::int32_t> labels(n * attributes);
  for (std::size_t i = 0; i < n; ++i) {
    labels[i * attributes] = static_cast<std::int32_t>(i % 4);
    labels[i * attributes + 1] = static_cast<std::int32_t>((i / 4) % 3);
    labels[i * attributes + 2] = i < 8 ? 9 : 0;
    for (std::size_t j = 0; j < dimension; ++j) {
      const float group = static_cast<float>(labels[i * attributes]) * 2.0F;
      vectors[i * dimension + j] =
          group + 0.01F * static_cast<float>(i) +
          0.05F * std::sin(static_cast<float>((i + 1) * (j + 1)));
    }
  }

  langer::BuildParameters build;
  build.num_clusters = 16;
  build.clustering_iterations = 8;
  build.max_degree = 6;
  build.construction_ef = 12;

  auto index = langer::Index::build(
      vectors, n, dimension, labels, attributes, build);

  const std::size_t query_id = 37;
  std::vector<float> query(
      vectors.begin() + static_cast<std::ptrdiff_t>(query_id * dimension),
      vectors.begin() + static_cast<std::ptrdiff_t>((query_id + 1) * dimension));

  langer::SearchParameters search;
  search.top_k = 5;
  search.ef_search = index.num_clusters();
  search.cluster_scan_budget = index.num_clusters();

  const auto single = index.search(query, {{0, labels[query_id * attributes]}}, search);
  const auto multiple = index.search(
      query,
      {{0, labels[query_id * attributes]},
       {1, labels[query_id * attributes + 1]}},
      search);

  if (single.empty() || multiple.empty()) {
    std::cerr << "LANGER smoke query returned no result\n";
    return 1;
  }
  std::cout << "LANGER index: " << index.num_clusters() << " centroids, "
            << index.num_value_graphs() << " value-specific graphs\n";
  std::cout << "single-predicate top result: " << single.front().id
            << ", distance=" << single.front().distance << '\n';
  std::cout << "multi-predicate top result: " << multiple.front().id
            << ", distance=" << multiple.front().distance << '\n';
  return 0;
}
