#include <langer/langer.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
std::pair<std::vector<T>, std::size_t> read_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<T> values;
  std::size_t width = 0;
  std::size_t rows = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream stream(line);
    std::string field;
    std::size_t row_width = 0;
    while (std::getline(stream, field, ',')) {
      std::stringstream scalar(field);
      T value{};
      scalar >> value;
      if (!scalar || !scalar.eof()) {
        throw std::runtime_error("invalid CSV value in " + path);
      }
      values.push_back(value);
      ++row_width;
    }
    if (row_width == 0 || (width != 0 && row_width != width)) {
      throw std::runtime_error("inconsistent CSV row width in " + path);
    }
    width = row_width;
    ++rows;
  }
  if (rows == 0) {
    throw std::runtime_error("empty CSV file: " + path);
  }
  return {std::move(values), width};
}

std::vector<float> parse_query(const std::string& text) {
  std::stringstream stream(text);
  std::string field;
  std::vector<float> query;
  while (std::getline(stream, field, ',')) {
    query.push_back(std::stof(field));
  }
  return query;
}

std::vector<langer::Predicate> parse_predicates(const std::string& text) {
  std::stringstream stream(text);
  std::string item;
  std::vector<langer::Predicate> predicates;
  while (std::getline(stream, item, ',')) {
    const auto equals = item.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("predicate must use attribute=value");
    }
    predicates.push_back(
        {static_cast<std::size_t>(std::stoull(item.substr(0, equals))),
         static_cast<std::int32_t>(std::stol(item.substr(equals + 1)))});
  }
  return predicates;
}

void usage(const char* program) {
  std::cerr
      << "Build: " << program
      << " build vectors.csv labels.csv index.langer num_clusters\n"
      << "Query: " << program
      << " query index.langer query_csv predicates top_k ef_search scan_clusters\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "build") {
      if (argc != 6) {
        usage(argv[0]);
        return 2;
      }
      auto vectors = read_csv<float>(argv[2]);
      auto labels = read_csv<std::int32_t>(argv[3]);
      const std::size_t n = vectors.first.size() / vectors.second;
      if (labels.first.size() / labels.second != n) {
        throw std::runtime_error("vectors and labels must have the same row count");
      }
      langer::BuildParameters parameters;
      parameters.num_clusters = std::stoull(argv[5]);
      auto index = langer::Index::build(vectors.first, n, vectors.second,
                                        labels.first, labels.second, parameters);
      index.save(argv[4]);
      std::cout << "saved " << argv[4] << " with " << index.num_clusters()
                << " centroids and " << index.num_value_graphs()
                << " value-specific graphs\n";
      return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "query") {
      if (argc != 8) {
        usage(argv[0]);
        return 2;
      }
      auto index = langer::Index::load(argv[2]);
      langer::SearchParameters parameters;
      parameters.top_k = std::stoull(argv[5]);
      parameters.ef_search = std::stoull(argv[6]);
      parameters.cluster_scan_budget = std::stoull(argv[7]);
      const auto result = index.search(
          parse_query(argv[3]), parse_predicates(argv[4]), parameters);
      std::cout << "vector_id,distance\n";
      for (const auto& hit : result) {
        std::cout << hit.id << ',' << hit.distance << '\n';
      }
      return 0;
    }

    usage(argv[0]);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "LANGER error: " << error.what() << '\n';
    return 1;
  }
}
