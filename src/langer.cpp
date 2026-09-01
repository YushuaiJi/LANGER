#include <langer/langer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace langer {
namespace {

constexpr std::array<char, 8> kFileMagic = {'L', 'A', 'N', 'G', 'E', 'R', '0', '1'};

float squared_l2(const float* left, const float* right, std::size_t dimension) {
  float distance = 0.0F;
  for (std::size_t j = 0; j < dimension; ++j) {
    const float delta = left[j] - right[j];
    distance += delta * delta;
  }
  return distance;
}

struct LayerCandidate {
  CentroidId id = 0;
  float similarity = 0.0F;
  float lower = 0.0F;
  float upper = 0.0F;
  float distance = 0.0F;
  std::size_t layer = 1;
};

class LayerTracker {
 public:
  explicit LayerTracker(const std::vector<LayerCandidate>& candidates) {
    coordinates_.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      coordinates_.push_back(candidate.lower);
    }
    std::sort(coordinates_.begin(), coordinates_.end());
    coordinates_.erase(
        std::unique(coordinates_.begin(), coordinates_.end()), coordinates_.end());
    tree_.assign(coordinates_.size() + 1, 0);
  }

  std::size_t max_layer_at_or_above(float threshold) const {
    const auto iterator =
        std::lower_bound(coordinates_.begin(), coordinates_.end(), threshold);
    if (iterator == coordinates_.end()) {
      return 0;
    }
    std::size_t position =
        coordinates_.size() - static_cast<std::size_t>(iterator - coordinates_.begin());
    std::size_t answer = 0;
    while (position > 0) {
      answer = std::max(answer, tree_[position]);
      position -= position & (~position + 1);
    }
    return answer;
  }

  void update(float lower, std::size_t layer) {
    const auto iterator =
        std::lower_bound(coordinates_.begin(), coordinates_.end(), lower);
    std::size_t position =
        coordinates_.size() - static_cast<std::size_t>(iterator - coordinates_.begin());
    while (position < tree_.size()) {
      tree_[position] = std::max(tree_[position], layer);
      position += position & (~position + 1);
    }
  }

 private:
  std::vector<float> coordinates_;
  std::vector<std::size_t> tree_;
};

std::vector<CentroidId> oppl_select(std::vector<LayerCandidate> candidates,
                                    std::size_t budget) {
  if (candidates.empty() || budget == 0) {
    return {};
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const LayerCandidate& left, const LayerCandidate& right) {
              if (std::abs(left.similarity - right.similarity) > 1.0e-12F) {
                return left.similarity > right.similarity;
              }
              if (left.distance != right.distance) {
                return left.distance < right.distance;
              }
              return left.id < right.id;
            });

  LayerTracker tracker(candidates);
  std::size_t begin = 0;
  while (begin < candidates.size()) {
    std::size_t end = begin + 1;
    while (end < candidates.size() &&
           std::abs(candidates[end].similarity - candidates[begin].similarity) <=
               1.0e-12F) {
      ++end;
    }
    for (std::size_t i = begin; i < end; ++i) {
      candidates[i].layer =
          tracker.max_layer_at_or_above(candidates[i].upper) + 1;
    }
    for (std::size_t i = begin; i < end; ++i) {
      tracker.update(candidates[i].lower, candidates[i].layer);
    }
    begin = end;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const LayerCandidate& left, const LayerCandidate& right) {
              if (left.layer != right.layer) {
                return left.layer < right.layer;
              }
              if (left.distance != right.distance) {
                return left.distance < right.distance;
              }
              return left.id < right.id;
            });
  const std::size_t selected = std::min(budget, candidates.size());
  std::vector<CentroidId> result;
  result.reserve(selected);
  for (std::size_t i = 0; i < selected; ++i) {
    result.push_back(candidates[i].id);
  }
  return result;
}

std::pair<float, float> score_interval(std::uint32_t cluster_size,
                                       std::uint32_t support,
                                       float average_cluster_size) {
  if (cluster_size == 0 || average_cluster_size <= 0.0F) {
    return {0.0F, 0.0F};
  }
  const float frequency =
      static_cast<float>(support) / static_cast<float>(cluster_size);
  const float balance = std::min(
      1.0F, static_cast<float>(cluster_size) / average_cluster_size);
  return {balance * balance * frequency, balance * frequency};
}

template <typename T>
void write_scalar(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!output) {
    throw std::runtime_error("failed while writing LANGER index");
  }
}

template <typename T>
T read_scalar(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("truncated LANGER index");
  }
  return value;
}

template <typename T>
void write_vector(std::ostream& output, const std::vector<T>& values) {
  const std::uint64_t size = static_cast<std::uint64_t>(values.size());
  write_scalar(output, size);
  if (!values.empty()) {
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(sizeof(T) * values.size()));
    if (!output) {
      throw std::runtime_error("failed while writing LANGER index array");
    }
  }
}

template <typename T>
std::vector<T> read_vector(std::istream& input) {
  const auto size = read_scalar<std::uint64_t>(input);
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                        std::max<std::size_t>(1, sizeof(T)))) {
    throw std::runtime_error("invalid array size in LANGER index");
  }
  std::vector<T> values(static_cast<std::size_t>(size));
  if (!values.empty()) {
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(sizeof(T) * values.size()));
    if (!input) {
      throw std::runtime_error("truncated LANGER index array");
    }
  }
  return values;
}

bool predicate_less(const Predicate& left, const Predicate& right) {
  return std::tie(left.attribute, left.value) <
         std::tie(right.attribute, right.value);
}

}  // namespace

std::size_t PredicateHash::operator()(const Predicate& predicate) const noexcept {
  const auto left = std::hash<std::size_t>{}(predicate.attribute);
  const auto right = std::hash<std::int32_t>{}(predicate.value);
  return left ^ (right + 0x9e3779b9U + (left << 6U) + (left >> 2U));
}

Index Index::build(const std::vector<float>& vectors,
                   std::size_t num_vectors,
                   std::size_t dimension,
                   const std::vector<std::int32_t>& labels,
                   std::size_t num_attributes,
                   const BuildParameters& parameters) {
  if (num_vectors == 0 || dimension == 0 || num_attributes == 0) {
    throw std::invalid_argument("LANGER requires non-empty vectors and labels");
  }
  if (vectors.size() != num_vectors * dimension ||
      labels.size() != num_vectors * num_attributes) {
    throw std::invalid_argument("row-major vector or label array has wrong size");
  }
  if (!(parameters.selectivity_cutoff >= 0.0F &&
        parameters.selectivity_cutoff <= 1.0F) ||
      parameters.max_degree == 0 || parameters.construction_ef == 0 ||
      parameters.entry_point_count == 0) {
    throw std::invalid_argument("invalid LANGER build parameters");
  }
  if (num_vectors > std::numeric_limits<VectorId>::max()) {
    throw std::invalid_argument("too many vectors for 32-bit LANGER vector IDs");
  }

  Index index;
  index.num_vectors_ = num_vectors;
  index.dimension_ = dimension;
  index.num_attributes_ = num_attributes;
  index.build_parameters_ = parameters;
  index.vectors_ = vectors;
  index.labels_ = labels;

  index.feature_mean_.assign(dimension, 0.0F);
  index.feature_stddev_.assign(dimension, 1.0F);
  std::vector<float> scaled(vectors.size());
  for (std::size_t j = 0; j < dimension; ++j) {
    double sum = 0.0;
    for (std::size_t i = 0; i < num_vectors; ++i) {
      sum += vectors[i * dimension + j];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(num_vectors));
    index.feature_mean_[j] = mean;
    double variance = 0.0;
    for (std::size_t i = 0; i < num_vectors; ++i) {
      const double delta = static_cast<double>(vectors[i * dimension + j]) - mean;
      variance += delta * delta;
    }
    float stddev = static_cast<float>(
        std::sqrt(variance / static_cast<double>(num_vectors)));
    if (stddev < 1.0e-6F) {
      stddev = 1.0F;
    }
    index.feature_stddev_[j] = stddev;
    for (std::size_t i = 0; i < num_vectors; ++i) {
      scaled[i * dimension + j] =
          (vectors[i * dimension + j] - mean) / stddev;
    }
  }

  std::unordered_map<Predicate, std::uint32_t, PredicateHash> global_counts;
  for (std::size_t i = 0; i < num_vectors; ++i) {
    for (std::size_t attribute = 0; attribute < num_attributes; ++attribute) {
      ++global_counts[{attribute, labels[i * num_attributes + attribute]}];
    }
  }
  std::vector<Predicate> all_predicates;
  all_predicates.reserve(global_counts.size());
  for (const auto& entry : global_counts) {
    all_predicates.push_back(entry.first);
  }
  std::sort(all_predicates.begin(), all_predicates.end(), predicate_less);

  std::vector<Predicate> rare_predicates;
  for (const auto& predicate : all_predicates) {
    const float frequency = static_cast<float>(global_counts.at(predicate)) /
                            static_cast<float>(num_vectors);
    if (frequency < parameters.selectivity_cutoff) {
      rare_predicates.push_back(predicate);
    }
  }
  std::unordered_map<Predicate, int, PredicateHash> rare_ids;
  for (std::size_t i = 0; i < rare_predicates.size(); ++i) {
    rare_ids[rare_predicates[i]] = static_cast<int>(i);
  }

  struct ExpandedPoint {
    VectorId id;
    int rare_id;
  };
  std::vector<ExpandedPoint> expanded;
  for (std::size_t i = 0; i < num_vectors; ++i) {
    bool tagged = false;
    for (std::size_t attribute = 0; attribute < num_attributes; ++attribute) {
      const Predicate predicate{attribute,
                                labels[i * num_attributes + attribute]};
      const auto iterator = rare_ids.find(predicate);
      if (iterator != rare_ids.end()) {
        expanded.push_back({static_cast<VectorId>(i), iterator->second});
        tagged = true;
      }
    }
    if (!tagged) {
      expanded.push_back({static_cast<VectorId>(i), -1});
    }
  }

  std::size_t k = parameters.num_clusters;
  if (k == 0) {
    k = std::max<std::size_t>(1, num_vectors / 1000);
  }
  k = std::min(k, expanded.size());
  if (k > std::numeric_limits<CentroidId>::max()) {
    throw std::invalid_argument("too many centroids for 32-bit LANGER IDs");
  }
  index.build_parameters_.num_clusters = k;

  std::mt19937_64 generator(parameters.random_seed);
  std::vector<std::size_t> shuffled(expanded.size());
  std::iota(shuffled.begin(), shuffled.end(), 0);
  std::shuffle(shuffled.begin(), shuffled.end(), generator);
  index.centroids_.assign(k * dimension, 0.0F);
  for (std::size_t centroid = 0; centroid < k; ++centroid) {
    const auto id = expanded[shuffled[centroid]].id;
    std::copy_n(scaled.data() + static_cast<std::size_t>(id) * dimension,
                dimension, index.centroids_.data() + centroid * dimension);
  }

  std::vector<CentroidId> assignments(expanded.size(), 0);
  auto assign_by_distance = [&]() {
    for (std::size_t e = 0; e < expanded.size(); ++e) {
      const float* point = scaled.data() +
                           static_cast<std::size_t>(expanded[e].id) * dimension;
      float best_distance = std::numeric_limits<float>::infinity();
      CentroidId best = 0;
      for (std::size_t centroid = 0; centroid < k; ++centroid) {
        const float distance = squared_l2(
            point, index.centroids_.data() + centroid * dimension, dimension);
        if (distance < best_distance) {
          best_distance = distance;
          best = static_cast<CentroidId>(centroid);
        }
      }
      assignments[e] = best;
    }
  };
  assign_by_distance();

  double total_scaled_square = 0.0;
  for (float value : scaled) {
    total_scaled_square += static_cast<double>(value) * value;
  }
  const float sigma_squared = static_cast<float>(
      total_scaled_square / static_cast<double>(scaled.size()));
  std::vector<float> rare_frequency(rare_predicates.size(), 1.0F);
  for (std::size_t r = 0; r < rare_predicates.size(); ++r) {
    rare_frequency[r] =
        static_cast<float>(global_counts.at(rare_predicates[r])) /
        static_cast<float>(num_vectors);
  }

  const std::size_t iterations =
      std::max<std::size_t>(1, parameters.clustering_iterations);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::vector<std::uint32_t> cluster_entries(k, 0);
    std::vector<std::vector<std::uint32_t>> rare_counts(
        k, std::vector<std::uint32_t>(rare_predicates.size(), 0));
    for (std::size_t e = 0; e < expanded.size(); ++e) {
      const auto centroid = assignments[e];
      ++cluster_entries[centroid];
      if (expanded[e].rare_id >= 0) {
        ++rare_counts[centroid][static_cast<std::size_t>(expanded[e].rare_id)];
      }
    }
    std::vector<int> dominant_rare(k, -1);
    std::vector<float> dominant_probability(k, 0.0F);
    for (std::size_t centroid = 0; centroid < k; ++centroid) {
      for (std::size_t r = 0; r < rare_predicates.size(); ++r) {
        const float probability = cluster_entries[centroid] == 0
                                      ? 0.0F
                                      : static_cast<float>(rare_counts[centroid][r]) /
                                            static_cast<float>(cluster_entries[centroid]);
        if (probability > dominant_probability[centroid]) {
          dominant_probability[centroid] = probability;
          dominant_rare[centroid] = static_cast<int>(r);
        }
      }
    }

    std::vector<CentroidId> next_assignments(assignments.size(), 0);
    for (std::size_t e = 0; e < expanded.size(); ++e) {
      const float* point = scaled.data() +
                           static_cast<std::size_t>(expanded[e].id) * dimension;
      float best_objective = std::numeric_limits<float>::infinity();
      CentroidId best = 0;
      for (std::size_t centroid = 0; centroid < k; ++centroid) {
        float objective = squared_l2(
            point, index.centroids_.data() + centroid * dimension, dimension);
        if (expanded[e].rare_id >= 0 && cluster_entries[centroid] > 0) {
          const std::size_t r = static_cast<std::size_t>(expanded[e].rare_id);
          const float probability =
              static_cast<float>(rare_counts[centroid][r]) /
              static_cast<float>(cluster_entries[centroid]);
          objective -= sigma_squared * probability /
                       std::max(rare_frequency[r], 1.0e-12F);
          if (dominant_rare[centroid] >= 0 &&
              dominant_rare[centroid] != expanded[e].rare_id) {
            objective += sigma_squared * dominant_probability[centroid];
          }
        }
        if (objective < best_objective) {
          best_objective = objective;
          best = static_cast<CentroidId>(centroid);
        }
      }
      next_assignments[e] = best;
    }

    std::vector<double> sums(k * dimension, 0.0);
    std::vector<std::uint32_t> counts(k, 0);
    for (std::size_t e = 0; e < expanded.size(); ++e) {
      const auto centroid = next_assignments[e];
      ++counts[centroid];
      const float* point = scaled.data() +
                           static_cast<std::size_t>(expanded[e].id) * dimension;
      for (std::size_t j = 0; j < dimension; ++j) {
        sums[static_cast<std::size_t>(centroid) * dimension + j] += point[j];
      }
    }
    for (std::size_t centroid = 0; centroid < k; ++centroid) {
      if (counts[centroid] == 0) {
        const auto id = expanded[shuffled[(iteration + centroid) % expanded.size()]].id;
        std::copy_n(scaled.data() + static_cast<std::size_t>(id) * dimension,
                    dimension, index.centroids_.data() + centroid * dimension);
      } else {
        for (std::size_t j = 0; j < dimension; ++j) {
          index.centroids_[centroid * dimension + j] = static_cast<float>(
              sums[centroid * dimension + j] / counts[centroid]);
        }
      }
    }
    const bool converged = next_assignments == assignments;
    assignments.swap(next_assignments);
    if (converged) {
      break;
    }
  }

  std::vector<std::unordered_set<VectorId>> posting_sets(k);
  for (std::size_t e = 0; e < expanded.size(); ++e) {
    posting_sets[assignments[e]].insert(expanded[e].id);
  }
  index.posting_lists_.resize(k);
  index.cluster_sizes_.resize(k, 0);
  index.cluster_predicate_counts_.resize(k);
  for (std::size_t centroid = 0; centroid < k; ++centroid) {
    auto& posting = index.posting_lists_[centroid];
    posting.assign(posting_sets[centroid].begin(), posting_sets[centroid].end());
    std::sort(posting.begin(), posting.end());
    index.cluster_sizes_[centroid] =
        static_cast<std::uint32_t>(posting.size());
    auto& counts = index.cluster_predicate_counts_[centroid];
    for (VectorId id : posting) {
      for (std::size_t attribute = 0; attribute < num_attributes; ++attribute) {
        ++counts[{attribute,
                  labels[static_cast<std::size_t>(id) * num_attributes + attribute]}];
      }
    }
  }

  const float average_cluster_size = static_cast<float>(
      std::accumulate(index.cluster_sizes_.begin(), index.cluster_sizes_.end(),
                      std::uint64_t{0})) /
                                     static_cast<float>(k);
  auto support = [&](CentroidId centroid, const Predicate& predicate) {
    const auto& counts = index.cluster_predicate_counts_[centroid];
    const auto iterator = counts.find(predicate);
    return iterator == counts.end() ? std::uint32_t{0} : iterator->second;
  };
  auto centroid_distance = [&](CentroidId left, CentroidId right) {
    return squared_l2(index.centroids_.data() +
                          static_cast<std::size_t>(left) * dimension,
                      index.centroids_.data() +
                          static_cast<std::size_t>(right) * dimension,
                      dimension);
  };

  for (const auto& predicate : all_predicates) {
    ValueGraph graph;
    graph.adjacency.resize(k);
    std::vector<CentroidId> centroid_order(k);
    std::iota(centroid_order.begin(), centroid_order.end(), CentroidId{0});
    std::sort(centroid_order.begin(), centroid_order.end(),
              [&](CentroidId left, CentroidId right) {
                const auto left_support = support(left, predicate);
                const auto right_support = support(right, predicate);
                if (left_support != right_support) {
                  return left_support > right_support;
                }
                if (index.cluster_sizes_[left] != index.cluster_sizes_[right]) {
                  return index.cluster_sizes_[left] > index.cluster_sizes_[right];
                }
                return left < right;
              });
    const std::size_t entry_count =
        std::min(parameters.entry_point_count, centroid_order.size());
    graph.entry_centroids.assign(centroid_order.begin(),
                                 centroid_order.begin() + entry_count);

    std::vector<bool> inserted(k, false);
    std::vector<CentroidId> insertion_order = graph.entry_centroids;
    for (CentroidId centroid : centroid_order) {
      if (std::find(insertion_order.begin(), insertion_order.end(), centroid) ==
          insertion_order.end()) {
        insertion_order.push_back(centroid);
      }
    }

    auto interval_for = [&](CentroidId centroid) {
      return score_interval(index.cluster_sizes_[centroid],
                            support(centroid, predicate), average_cluster_size);
    };
    auto prune = [&](CentroidId centroid) {
      auto neighbors = graph.adjacency[centroid];
      if (neighbors.size() <= parameters.max_degree) {
        return;
      }
      std::vector<LayerCandidate> candidates;
      candidates.reserve(neighbors.size());
      for (CentroidId neighbor : neighbors) {
        const float distance = centroid_distance(centroid, neighbor);
        const auto interval = interval_for(neighbor);
        candidates.push_back({neighbor, 1.0F / (1.0F + distance),
                              interval.first, interval.second, distance, 1});
      }
      const auto selected = oppl_select(candidates, parameters.max_degree);
      const std::unordered_set<CentroidId> keep(selected.begin(), selected.end());
      for (CentroidId neighbor : neighbors) {
        if (keep.find(neighbor) == keep.end()) {
          auto& reverse = graph.adjacency[neighbor];
          reverse.erase(std::remove(reverse.begin(), reverse.end(), centroid),
                        reverse.end());
        }
      }
      graph.adjacency[centroid] = selected;
    };
    auto add_edge = [&](CentroidId left, CentroidId right) {
      if (left == right) {
        return;
      }
      auto& left_neighbors = graph.adjacency[left];
      auto& right_neighbors = graph.adjacency[right];
      if (std::find(left_neighbors.begin(), left_neighbors.end(), right) ==
          left_neighbors.end()) {
        left_neighbors.push_back(right);
      }
      if (std::find(right_neighbors.begin(), right_neighbors.end(), left) ==
          right_neighbors.end()) {
        right_neighbors.push_back(left);
      }
      prune(left);
      prune(right);
    };

    std::size_t inserted_count = 0;
    for (CentroidId centroid : insertion_order) {
      if (inserted[centroid]) {
        continue;
      }
      if (inserted_count == 0) {
        inserted[centroid] = true;
        ++inserted_count;
        continue;
      }

      using QueueItem = std::pair<float, CentroidId>;
      std::priority_queue<QueueItem, std::vector<QueueItem>,
                          std::greater<QueueItem>> queue;
      std::vector<bool> visited(k, false);
      for (CentroidId entry : graph.entry_centroids) {
        if (inserted[entry] && !visited[entry]) {
          visited[entry] = true;
          queue.push({centroid_distance(centroid, entry), entry});
        }
      }
      if (queue.empty()) {
        for (CentroidId candidate = 0; candidate < k; ++candidate) {
          if (inserted[candidate]) {
            visited[candidate] = true;
            queue.push({centroid_distance(centroid, candidate), candidate});
            break;
          }
        }
      }
      std::vector<CentroidId> candidates;
      const std::size_t candidate_budget =
          std::min(parameters.construction_ef, inserted_count);
      while (!queue.empty() && candidates.size() < candidate_budget) {
        const auto current = queue.top();
        queue.pop();
        candidates.push_back(current.second);
        for (CentroidId neighbor : graph.adjacency[current.second]) {
          if (inserted[neighbor] && !visited[neighbor]) {
            visited[neighbor] = true;
            queue.push({centroid_distance(centroid, neighbor), neighbor});
          }
        }
      }
      if (candidates.size() < candidate_budget) {
        std::vector<QueueItem> remaining;
        for (CentroidId candidate = 0; candidate < k; ++candidate) {
          if (inserted[candidate] && !visited[candidate]) {
            remaining.push_back(
                {centroid_distance(centroid, candidate), candidate});
          }
        }
        std::sort(remaining.begin(), remaining.end());
        for (const auto& item : remaining) {
          if (candidates.size() >= candidate_budget) {
            break;
          }
          candidates.push_back(item.second);
        }
      }

      std::vector<LayerCandidate> layer_candidates;
      layer_candidates.reserve(candidates.size());
      for (CentroidId candidate : candidates) {
        const float distance = centroid_distance(centroid, candidate);
        const auto interval = interval_for(candidate);
        layer_candidates.push_back(
            {candidate, 1.0F / (1.0F + distance), interval.first,
             interval.second, distance, 1});
      }
      auto selected = oppl_select(layer_candidates, parameters.max_degree);
      if (selected.empty() && !candidates.empty()) {
        selected.push_back(candidates.front());
      }
      inserted[centroid] = true;
      ++inserted_count;
      for (CentroidId neighbor : selected) {
        add_edge(centroid, neighbor);
      }
    }
    index.graphs_.emplace(predicate, std::move(graph));
  }

  return index;
}

std::vector<SearchResult> Index::search(
    const std::vector<float>& query,
    const std::vector<Predicate>& predicates,
    const SearchParameters& parameters) const {
  if (query.size() != dimension_) {
    throw std::invalid_argument("query dimension does not match LANGER index");
  }
  if (predicates.empty() || parameters.top_k == 0 || parameters.ef_search == 0 ||
      parameters.cluster_scan_budget == 0) {
    return {};
  }
  for (const auto& predicate : predicates) {
    if (predicate.attribute >= num_attributes_) {
      throw std::invalid_argument("predicate attribute is outside label schema");
    }
    if (graphs_.find(predicate) == graphs_.end()) {
      return {};
    }
  }

  std::vector<float> scaled_query(dimension_);
  for (std::size_t j = 0; j < dimension_; ++j) {
    scaled_query[j] = (query[j] - feature_mean_[j]) / feature_stddev_[j];
  }
  auto centroid_distance = [&](CentroidId centroid) {
    return squared_l2(scaled_query.data(),
                      centroids_.data() +
                          static_cast<std::size_t>(centroid) * dimension_,
                      dimension_);
  };

  auto traverse = [&](const ValueGraph& graph) {
    using QueueItem = std::pair<float, CentroidId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<QueueItem>> queue;
    std::vector<bool> visited(num_clusters(), false);
    for (CentroidId entry : graph.entry_centroids) {
      if (!visited[entry]) {
        visited[entry] = true;
        queue.push({centroid_distance(entry), entry});
      }
    }
    std::vector<CentroidId> result;
    const std::size_t budget = std::min(parameters.ef_search, num_clusters());
    while (!queue.empty() && result.size() < budget) {
      const auto current = queue.top();
      queue.pop();
      result.push_back(current.second);
      for (CentroidId neighbor : graph.adjacency[current.second]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push({centroid_distance(neighbor), neighbor});
        }
      }
    }
    if (result.size() < budget) {
      std::vector<QueueItem> remaining;
      for (CentroidId centroid = 0; centroid < num_clusters(); ++centroid) {
        if (!visited[centroid]) {
          remaining.push_back({centroid_distance(centroid), centroid});
        }
      }
      std::sort(remaining.begin(), remaining.end());
      for (const auto& item : remaining) {
        if (result.size() >= budget) {
          break;
        }
        result.push_back(item.second);
      }
    }
    return result;
  };

  std::vector<CentroidId> scan_order;
  if (predicates.size() == 1) {
    scan_order = traverse(graphs_.at(predicates.front()));
    std::sort(scan_order.begin(), scan_order.end(),
              [&](CentroidId left, CentroidId right) {
                return std::make_tuple(centroid_distance(left), left) <
                       std::make_tuple(centroid_distance(right), right);
              });
  } else {
    std::unordered_set<CentroidId> union_set;
    for (const auto& predicate : predicates) {
      for (CentroidId centroid : traverse(graphs_.at(predicate))) {
        union_set.insert(centroid);
      }
    }
    const float average_cluster_size = static_cast<float>(
        std::accumulate(cluster_sizes_.begin(), cluster_sizes_.end(),
                        std::uint64_t{0})) /
                                       static_cast<float>(num_clusters());
    std::vector<LayerCandidate> candidates;
    candidates.reserve(union_set.size());
    for (CentroidId centroid : union_set) {
      std::uint32_t minimum_support =
          std::numeric_limits<std::uint32_t>::max();
      for (const auto& predicate : predicates) {
        const auto& counts = cluster_predicate_counts_[centroid];
        const auto iterator = counts.find(predicate);
        const std::uint32_t count =
            iterator == counts.end() ? 0U : iterator->second;
        minimum_support = std::min(minimum_support, count);
      }
      const auto interval = score_interval(
          cluster_sizes_[centroid], minimum_support, average_cluster_size);
      const float distance = centroid_distance(centroid);
      candidates.push_back({centroid, 1.0F / (1.0F + distance),
                            interval.first, interval.second, distance, 1});
    }
    scan_order = oppl_select(candidates, parameters.cluster_scan_budget);
  }
  if (scan_order.size() > parameters.cluster_scan_budget) {
    scan_order.resize(parameters.cluster_scan_budget);
  }

  struct HeapResult {
    float distance;
    VectorId id;
    bool operator<(const HeapResult& other) const {
      return std::tie(distance, id) < std::tie(other.distance, other.id);
    }
  };
  std::priority_queue<HeapResult> heap;
  std::vector<bool> visited_vectors(num_vectors_, false);
  for (CentroidId centroid : scan_order) {
    for (VectorId id : posting_lists_[centroid]) {
      if (visited_vectors[id]) {
        continue;
      }
      visited_vectors[id] = true;
      bool qualified = true;
      for (const auto& predicate : predicates) {
        if (labels_[static_cast<std::size_t>(id) * num_attributes_ +
                    predicate.attribute] != predicate.value) {
          qualified = false;
          break;
        }
      }
      if (!qualified) {
        continue;
      }
      const float distance = squared_l2(
          query.data(), vectors_.data() + static_cast<std::size_t>(id) * dimension_,
          dimension_);
      if (heap.size() < parameters.top_k) {
        heap.push({distance, id});
      } else if (std::tie(distance, id) <
                 std::tie(heap.top().distance, heap.top().id)) {
        heap.pop();
        heap.push({distance, id});
      }
    }
  }

  std::vector<SearchResult> result;
  result.reserve(heap.size());
  while (!heap.empty()) {
    result.push_back({heap.top().id, heap.top().distance});
    heap.pop();
  }
  std::sort(result.begin(), result.end(),
            [](const SearchResult& left, const SearchResult& right) {
              return std::tie(left.distance, left.id) <
                     std::tie(right.distance, right.id);
            });
  return result;
}

std::size_t Index::max_observed_out_degree() const noexcept {
  std::size_t maximum = 0;
  for (const auto& entry : graphs_) {
    for (const auto& neighbors : entry.second.adjacency) {
      maximum = std::max(maximum, neighbors.size());
    }
  }
  return maximum;
}

void Index::save(const std::string& path) const {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create LANGER index: " + path);
  }
  output.write(kFileMagic.data(), static_cast<std::streamsize>(kFileMagic.size()));
  write_scalar(output, static_cast<std::uint64_t>(num_vectors_));
  write_scalar(output, static_cast<std::uint64_t>(dimension_));
  write_scalar(output, static_cast<std::uint64_t>(num_attributes_));
  write_scalar(output, static_cast<std::uint64_t>(build_parameters_.num_clusters));
  write_scalar(output,
               static_cast<std::uint64_t>(build_parameters_.clustering_iterations));
  write_scalar(output, build_parameters_.selectivity_cutoff);
  write_scalar(output, static_cast<std::uint64_t>(build_parameters_.max_degree));
  write_scalar(output,
               static_cast<std::uint64_t>(build_parameters_.construction_ef));
  write_scalar(output,
               static_cast<std::uint64_t>(build_parameters_.entry_point_count));
  write_scalar(output, build_parameters_.random_seed);
  write_vector(output, vectors_);
  write_vector(output, labels_);
  write_vector(output, feature_mean_);
  write_vector(output, feature_stddev_);
  write_vector(output, centroids_);
  write_vector(output, cluster_sizes_);

  write_scalar(output, static_cast<std::uint64_t>(posting_lists_.size()));
  for (const auto& posting : posting_lists_) {
    write_vector(output, posting);
  }
  write_scalar(output,
               static_cast<std::uint64_t>(cluster_predicate_counts_.size()));
  for (const auto& counts : cluster_predicate_counts_) {
    write_scalar(output, static_cast<std::uint64_t>(counts.size()));
    for (const auto& count : counts) {
      write_scalar(output, static_cast<std::uint64_t>(count.first.attribute));
      write_scalar(output, count.first.value);
      write_scalar(output, count.second);
    }
  }
  write_scalar(output, static_cast<std::uint64_t>(graphs_.size()));
  for (const auto& entry : graphs_) {
    write_scalar(output, static_cast<std::uint64_t>(entry.first.attribute));
    write_scalar(output, entry.first.value);
    write_vector(output, entry.second.entry_centroids);
    write_scalar(output,
                 static_cast<std::uint64_t>(entry.second.adjacency.size()));
    for (const auto& neighbors : entry.second.adjacency) {
      write_vector(output, neighbors);
    }
  }
}

Index Index::load(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open LANGER index: " + path);
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kFileMagic) {
    throw std::runtime_error("unsupported LANGER index format");
  }

  Index index;
  index.num_vectors_ = static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.dimension_ = static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.num_attributes_ =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.num_clusters =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.clustering_iterations =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.selectivity_cutoff = read_scalar<float>(input);
  index.build_parameters_.max_degree =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.construction_ef =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.entry_point_count =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(input));
  index.build_parameters_.random_seed = read_scalar<std::uint64_t>(input);
  index.vectors_ = read_vector<float>(input);
  index.labels_ = read_vector<std::int32_t>(input);
  index.feature_mean_ = read_vector<float>(input);
  index.feature_stddev_ = read_vector<float>(input);
  index.centroids_ = read_vector<float>(input);
  index.cluster_sizes_ = read_vector<std::uint32_t>(input);

  const auto posting_count = read_scalar<std::uint64_t>(input);
  index.posting_lists_.resize(static_cast<std::size_t>(posting_count));
  for (auto& posting : index.posting_lists_) {
    posting = read_vector<VectorId>(input);
  }
  const auto count_table_size = read_scalar<std::uint64_t>(input);
  index.cluster_predicate_counts_.resize(
      static_cast<std::size_t>(count_table_size));
  for (auto& counts : index.cluster_predicate_counts_) {
    const auto count_size = read_scalar<std::uint64_t>(input);
    for (std::uint64_t i = 0; i < count_size; ++i) {
      const Predicate predicate{
          static_cast<std::size_t>(read_scalar<std::uint64_t>(input)),
          read_scalar<std::int32_t>(input)};
      counts.emplace(predicate, read_scalar<std::uint32_t>(input));
    }
  }
  const auto graph_count = read_scalar<std::uint64_t>(input);
  for (std::uint64_t g = 0; g < graph_count; ++g) {
    const Predicate predicate{
        static_cast<std::size_t>(read_scalar<std::uint64_t>(input)),
        read_scalar<std::int32_t>(input)};
    ValueGraph graph;
    graph.entry_centroids = read_vector<CentroidId>(input);
    const auto adjacency_size = read_scalar<std::uint64_t>(input);
    graph.adjacency.resize(static_cast<std::size_t>(adjacency_size));
    for (auto& neighbors : graph.adjacency) {
      neighbors = read_vector<CentroidId>(input);
    }
    index.graphs_.emplace(predicate, std::move(graph));
  }

  if (index.vectors_.size() != index.num_vectors_ * index.dimension_ ||
      index.labels_.size() != index.num_vectors_ * index.num_attributes_ ||
      index.feature_mean_.size() != index.dimension_ ||
      index.feature_stddev_.size() != index.dimension_ ||
      index.centroids_.size() != index.cluster_sizes_.size() * index.dimension_ ||
      index.posting_lists_.size() != index.cluster_sizes_.size() ||
      index.cluster_predicate_counts_.size() != index.cluster_sizes_.size()) {
    throw std::runtime_error("inconsistent LANGER index contents");
  }
  for (const auto& entry : index.graphs_) {
    if (entry.second.adjacency.size() != index.num_clusters()) {
      throw std::runtime_error("invalid value-specific graph in LANGER index");
    }
  }
  return index;
}

}  // namespace langer
