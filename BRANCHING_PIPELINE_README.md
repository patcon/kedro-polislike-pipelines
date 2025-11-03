# Branching Pipeline Feature

This document describes the new branching DAG pipeline feature that allows you to run multiple reducer-clusterer combinations efficiently by sharing intermediate results.

## Overview

The branching pipeline creates a single DAG where:
- **Preprocessing runs once** (download, deduplication, masking)
- **Each reducer runs once** (PCA, PaCMAP, UMAP, etc.)
- **Multiple clusterers run on each reducer's output** (BestKMeans, BestHDBSCANFlat, etc.)

This eliminates redundant computation by sharing intermediate results across multiple pipeline branches.

## Configuration

The branching pipeline is configured in `conf/base/parameters_experimental.yml`:

```yaml
branching_pipeline:
  shared_stages:
    imputer:
      name: SimpleImputer
      strategy: mean
    scaler:
      name: SparsityAwareScaler
      X_sparse: "input:masked_vote_matrix"
    filter:
      name: SampleMaskFilter
      mask: "input:participant_mask"

  reducers:
    - name: PCA
      n_components: 2
      random_state: ${globals:random_state}
    - name: PaCMAP
      n_components: 2
      n_neighbors: null
      random_state: ${globals:random_state}
    - name: UMAP
      n_components: 2
      n_neighbors: 15
      random_state: ${globals:random_state}

  clusterers:
    - name: BestKMeans
      k_bounds: [2, 5]
      random_state: ${globals:random_state}
    - name: BestHDBSCANFlat
      k_bounds: [2, 7]
      random_state: ${globals:random_state}
```

## Usage

### Run All Combinations

```bash
# Run all reducer-clusterer combinations (3 reducers × 2 clusterers = 6 combinations)
uv run python scripts/run_pipelines.py branching --params "polis_url=https://pol.is/report/r2dfw8eambusb8buvecjt"
```

### Run Specific Combinations with Tags

```bash
# Run only PCA + BestKMeans combination
uv run python scripts/run_pipelines.py branching --tags "pca_bestkmeans" --params "polis_url=https://pol.is/report/r2dfw8eambusb8buvecjt"

# Run all PCA combinations (PCA + BestKMeans, PCA + BestHDBSCANFlat)
uv run python scripts/run_pipelines.py branching --tags "pca" --params "polis_url=https://pol.is/report/r2dfw8eambusb8buvecjt"

# Run all BestKMeans combinations (PCA + BestKMeans, PaCMAP + BestKMeans, UMAP + BestKMeans)
uv run python scripts/run_pipelines.py branching --tags "bestkmeans" --params "polis_url=https://pol.is/report/r2dfw8eambusb8buvecjt"

# Run multiple specific combinations
uv run python scripts/run_pipelines.py branching --tags "pca_bestkmeans,umap_besthdbscanflat" --params "polis_url=https://pol.is/report/r2dfw8eambusb8buvecjt"
```

## Pipeline Structure

The branching pipeline creates the following DAG structure:

```
preprocessing → imputer → reducer_pca → scaler → filter → clusterer (PCA + BestKMeans)
                       ↘ reducer_pacmap → scaler → filter → clusterer (PaCMAP + BestKMeans)
                       ↘ reducer_umap → scaler → filter → clusterer (UMAP + BestKMeans)
                                                        ↘ clusterer (PCA + BestHDBSCANFlat)
                                                        ↘ clusterer (PaCMAP + BestHDBSCANFlat)
                                                        ↘ clusterer (UMAP + BestHDBSCANFlat)
```

## Available Tags

Each node in the pipeline is tagged for easy filtering:

- **Combination tags**: `pca_bestkmeans`, `pacmap_bestkmeans`, `umap_besthdbscanflat`, etc.
- **Reducer tags**: `pca`, `pacmap`, `umap`
- **Clusterer tags**: `bestkmeans`, `besthdbscanflat`
- **Stage tags**: `reducer`, `scaler`, `filter`, `clusterer`
- **Function tags**: `visualization`, `dataset`

## Outputs

Each combination produces the same outputs as individual pipelines:

- **Scatter plots**: Colored by cluster and participant ID
- **Red-Dwarf datasets**: `votes.parquet`, `projections.json`, `statements.json`, `meta.json`
- **Images**: Saved scatter plot images

Output files are organized by combination name:
```
data/{polis_id}/pca_bestkmeans/
data/{polis_id}/pacmap_bestkmeans/
data/{polis_id}/umap_besthdbscanflat/
...
```

## Benefits

1. **Efficiency**: Preprocessing and each reducer runs only once, regardless of how many clusterers use their output
2. **Flexibility**: Use tags to run only specific combinations you're interested in
3. **Scalability**: Easy to add new reducers or clusterers without duplicating computation
4. **Caching**: Kedro automatically caches intermediate results, so you can resume failed runs
5. **Visualization**: Full compatibility with Kedro-Viz for pipeline visualization

## Comparison with Individual Pipelines

| Approach | Preprocessing Runs | PCA Runs | Total Nodes | Flexibility |
|----------|-------------------|----------|-------------|-------------|
| Individual Pipelines | 6 times | 2 times | ~60 | Medium |
| Branching Pipeline | 1 time | 1 time | ~77 | High |

The branching pipeline creates more nodes but eliminates redundant computation, making it more efficient for running multiple combinations.