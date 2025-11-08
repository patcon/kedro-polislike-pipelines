# Pipeline Variants System

This document explains the new variants-based approach for managing pipeline combinations in the Kedro Polis Pipelines project.

## Overview

The variants system replaces the previous `enabled_combinations` approach with a more flexible and maintainable configuration structure. Instead of specifying individual combinations, you can now define named variants that group related pipeline configurations.

## Benefits

- **🧩 Cleaner Configuration**: Named variants with descriptions make it easier to understand what each configuration does
- **🚀 Singleton Support**: Easy definition of single combinations for CI/testing without matrix explosion
- **🎯 Selective Inclusion**: Control which estimators are included in the full matrix vs. available only for specific variants
- **📝 Custom Naming**: Override auto-generated names for better pipeline organization

## Configuration Structure

### Basic Variant Types

#### 1. Full Matrix Variant
```yaml
variants:
  full_matrix:
    description: "Full matrix expansion (all combinations with include_in_full_matrix=true)"
    active: false  # Disabled by default to avoid large matrix expansion
    # Uses all estimators where include_in_full_matrix=true (or unspecified)
```

#### 2. Singleton Variant
```yaml
variants:
  zero_pacmap_masked_bestkmeans:
    name: zero_pacmap_masked_bestkmeans  # Custom name for pipeline nodes
    active: true
    imputer: noop           # Single imputer
    reducer: pacmap_masked  # Single reducer  
    clusterer: bestkmeans   # Single clusterer
```

#### 3. List-Based Variant (Future)
```yaml
variants:
  pca_focused:
    description: "Focus on PCA with multiple imputers and clusterers"
    active: false
    imputers: [mean, median, knn5d]  # Multiple imputers
    reducers: [pca]                  # Single reducer
    clusterers: [bestkmeans, besthdbscan]  # Multiple clusterers
```

### Estimator Configuration

Each estimator can optionally specify `include_in_full_matrix`:

```yaml
imputers:
  - name: mean
    estimator: SimpleImputer
    strategy: mean
    # include_in_full_matrix: true (default)

  - name: noop
    estimator: NoOpTransformer
    include_in_full_matrix: false  # Only available for specific variants
```

## Usage Examples

### Example 1: CI/Testing Pipeline
```yaml
variants:
  ci_test:
    name: fast_test
    active: true
    imputer: mean
    reducer: pca
    clusterer: bestkmeans
```

This creates a single, fast-running pipeline perfect for CI checks.

### Example 2: Research Experiment
```yaml
variants:
  manifold_study:
    description: "Compare manifold learning methods"
    active: false
    imputers: [mean]
    reducers: [pacmap, localmap, umap]
    clusterers: [bestkmeans]
```

This creates 3 combinations to compare different dimensionality reduction methods.

### Example 3: Custom Analysis
```yaml
variants:
  raw_data_analysis:
    name: zero_pacmap_masked_bestkmeans
    active: true
    imputer: noop           # No imputation
    reducer: pacmap_masked  # Special PaCMAP with masked distance
    clusterer: bestkmeans
```

This creates a specialized pipeline for analyzing raw data without imputation.

## Migration from enabled_combinations

### Old Approach
```yaml
enabled_combinations:
  - imputer: SimpleImputer
    reducer: PCA
    clusterer: BestKMeans
  - imputer: KNNImputer
    reducer: UMAP
    clusterer: BestHDBSCANFlat
```

### New Approach
```yaml
variants:
  basic_comparison:
    description: "Compare basic imputation methods"
    active: true
    imputers: [mean, knn5d]
    reducers: [pca, umap]
    clusterers: [bestkmeans, besthdbscan]
```

## Implementation Details

### Pipeline Node Naming

- **Default naming**: `{imputer}_{reducer}_{clusterer}` (e.g., `mean_pca_bestkmeans`)
- **Custom naming**: Use the `name` field in singleton variants
- **Node suffixes**: `_clusterer_node`, `_save_meta_json`, `_create_scatter_plot`, etc.

### Variant Processing Order

1. **Variants take precedence** over `enabled_combinations`
2. **Active variants only** are processed (`active: true`)
3. **Multiple active variants** can coexist (combinations are merged)
4. **No active variants** falls back to legacy behavior

### Performance Considerations

- **Full matrix disabled by default** to prevent accidental large pipeline creation
- **Singleton variants** create minimal node count (typically 5-20 nodes)
- **List variants** create Cartesian products (use carefully)

## Best Practices

### 1. Use Descriptive Names
```yaml
variants:
  clustering_comparison:  # Clear purpose
    description: "Compare clustering algorithms on PCA-reduced data"
```

### 2. Control Matrix Inclusion
```yaml
reducers:
  - name: experimental_method
    estimator: ExperimentalReducer
    include_in_full_matrix: false  # Keep experimental methods out of full matrix
```

### 3. Organize by Purpose
```yaml
variants:
  # Production pipeline
  production:
    name: prod_pipeline
    active: true
    imputer: mean
    reducer: pca
    clusterer: bestkmeans

  # Research experiments
  research_manifold:
    active: false
    imputers: [mean]
    reducers: [pacmap, localmap, umap]
    clusterers: [bestkmeans]
```

### 4. Use Custom Names for Important Pipelines
```yaml
variants:
  main_analysis:
    name: polis_classic_v2  # Clear, meaningful name
    active: true
    imputer: mean
    reducer: pca
    clusterer: bestkmeans
```

## Troubleshooting

### Common Issues

1. **No nodes created**: Check that at least one variant is `active: true`
2. **Missing estimators**: Verify estimator names match those in the `imputers`/`reducers`/`clusterers` lists
3. **Unexpected combinations**: Check `include_in_full_matrix` settings on estimators

### Debugging

```python
# Check active variants
from kedro.config import OmegaConfigLoader
config_loader = OmegaConfigLoader(conf_source='conf', base_env='base', default_run_env='local')
params = config_loader['parameters']
variants = params.get('branching_pipeline', {}).get('variants', {})
active_variants = {k: v for k, v in variants.items() if v.get('active', False)}
print("Active variants:", active_variants)
```

## Future Enhancements

- **Conditional variants**: Activate variants based on environment or parameters
- **Variant inheritance**: Allow variants to extend other variants
- **Dynamic variants**: Generate variants programmatically
- **Variant validation**: Ensure estimator combinations are valid before pipeline creation