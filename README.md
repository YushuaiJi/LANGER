# LANGER

**LANGER** (Label-Aware Navigation Graph over Cluster Centroids) is a C++17
research implementation for label-filtered approximate nearest-neighbor search
(LANNS).

The implementation follows the current method design:

- **Label-Concentrating K-Means** groups vectors carrying the same
  low-frequency label value into fewer clusters.
- **Predicate-Aware Pareto Frontier (PAPF)** selects centroid neighbors using
  vector proximity and a label-aware centroid score interval.
- **One-Pass PAPF Layering (OPPL)** constructs the PAPF layers used by each
  value-specific centroid graph.
- **Minimal-Support Re-ranking (MSR)** merges and re-orders centroid candidates
  for queries containing multiple label predicates.

The repository contains implementation code and synthetic examples only. It
does not contain the manuscript, datasets, benchmark results, or private
cluster configuration.

## Requirements

- A C++17 compiler (AppleClang, Clang, or GCC)
- CMake 3.16 or newer

LANGER has no Python or third-party runtime dependency.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the self-contained example:

```bash
./build/langer_synthetic_smoke
```

Run the minimal test:

```bash
ctest --test-dir build --output-on-failure
```

## Build an index from CSV

`vectors.csv` must contain one floating-point vector per row. `labels.csv` must
contain one integer label value per attribute, with the same number of rows.

```bash
./build/langer_cli build vectors.csv labels.csv index.langer 100
```

The last argument is the number of coarse centroids. Use the paper default
`K = n / 1000` for the main experimental setting. The command writes a binary
index containing the vectors, labels, clusters, value-specific graphs, and all
metadata needed for search.

## Query an index

Pass the query vector and predicates as comma-separated strings. A predicate
`0=12` means that label attribute 0 must equal 12.

```bash
./build/langer_cli query index.langer \
  "0.12,0.05,-0.31,0.88" \
  "0=12,1=7" \
  10 100 32
```

The final three arguments are `top_k`, `ef_search`, and the maximum number of
clusters to scan. The command prints `vector_id,distance` pairs. A
single-predicate query traverses one value-specific graph. A multi-predicate
query traverses one graph per predicate and applies MSR before cluster scans.

## C++ API

```cpp
#include <langer/langer.hpp>

std::vector<float> vectors = /* row-major: n * dimension */;
std::vector<std::int32_t> labels = /* row-major: n * attributes */;

langer::BuildParameters build;
build.num_clusters = 100;
build.selectivity_cutoff = 0.01F;
build.max_degree = 20;
build.construction_ef = 200;

auto index = langer::Index::build(
    vectors, n, dimension, labels, attributes, build);
index.save("index.langer");

auto loaded = langer::Index::load("index.langer");
langer::SearchParameters search;
search.top_k = 10;
search.ef_search = 100;
search.cluster_scan_budget = 32;

auto result = loaded.search(
    query, {{0, 12}, {1, 7}}, search);
```

`BuildParameters::num_clusters = 0` selects `max(1, n / 1000)` automatically.
The paper defaults are `tau = 0.01`, `M = 20`, and construction `ef = 200`.

## Source layout

```text
include/langer/langer.hpp  Public C++ API
src/langer.cpp             LANGER construction, search, and persistence
examples/synthetic_smoke.cpp
examples/langer_cli.cpp
tests/test_langer.cpp
```

This is a readable CPU reference implementation of the current algorithms. It
is intended for method inspection and reproducible integration; reported paper
throughput should be reproduced with the corresponding experimental hardware,
datasets, compiler settings, and benchmark configuration.
