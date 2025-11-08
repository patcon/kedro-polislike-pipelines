# Kedro Polis-like Pipelines: Template

This project is an attempt to use Kedro to model [Polis](https://pol.is/home)-like data pipelines.

This repo builds and runs a set of polis-like pipelines via continuous integration, and publishes a small micro-site that exposes all the pipelines and data.

## Usage

You can fork this repo directly, or use it as a template repo for your own Polis conversations.

### Existing Pipeline Repos

- Blacksky: Community Guidelines. [about](https://bsky.app/profile/rude1.blacksky.team/post/3lxx52acerc2s)
  - [pipeline repo](https://github.com/patcon/kedro-polislike-pipelines-blacksky) | [micro-site](https://patcon.github.io/kedro-polislike-pipelines-blacksky/?types=parameters,datasets&pid=mean_localmap_bestkmeans&expandAllPipelines=false&sid=b9680c9f) | [data explorer 🔭](https://main--68c53b7909ee2fb48f1979dd.chromatic.com/iframe.html?args=kedroBaseUrl%3Ahttps__COLON____SLASH____SLASH__patcon__DOT__github__DOT__io__SLASH__kedro-polislike-pipelines-blacksky&globals=&id=components-app-kedro-mode--kedro-mode-with-animation&viewMode=story)
  - original polis: [report](https://assembly.blacksky.community/report/r9pnvme4e39uy5a3uptmr)
- Austrian Klimarat. [about](https://klimarat.org/)
  - Energy
    - [pipeline repo](https://github.com/patcon/kedro-polislike-pipeline-klimarat-energy) | [micro-site](https://patcon.github.io/kedro-polislike-pipeline-klimarat-energy/?types=parameters,datasets&pid=mean_localmap_bestkmeans&expandAllPipelines=false&sid=b9680c9f) | [data explorer 🔭](https://main--68c53b7909ee2fb48f1979dd.chromatic.com/iframe.html?args=kedroBaseUrl%3Ahttps__COLON____SLASH____SLASH__patcon__DOT__github__DOT__io__SLASH__kedro-polislike-pipeline-klimarat-energy&globals=&id=components-app-kedro-mode--kedro-mode-with-animation&viewMode=story)
    - original polis: [report](https://pol.is/report/r8nssrnnnf2bewvtd5f5h)
  - Mobility
    - [pipeline repo](https://github.com/patcon/kedro-polislike-pipelines-Klimarat-mobility) | [micro-site](https://patcon.github.io/kedro-polislike-pipelines-Klimarat-mobility/?types=parameters,datasets&pid=mean_localmap_bestkmeans&expandAllPipelines=false&sid=b9680c9f) | [data explorer 🔭](https://main--68c53b7909ee2fb48f1979dd.chromatic.com/iframe.html?args=kedroBaseUrl%3Ahttps__COLON____SLASH____SLASH__patcon__DOT__github__DOT__io__SLASH__kedro-polislike-pipelines-Klimarat-mobility&globals=&id=components-app-kedro-mode--kedro-mode-with-animation&viewMode=story)
    - original polis: [report](https://pol.is/report/r5bbmenm6nt3nnmf9dpvk)
- San Juan Islands Land Trust (testing geographic projections)
  - [pipeline repo](https://github.com/patcon/kedro-polislike-pipelines-san-juan-islands) | [micro-site]() | [data explorer 🔭](https://main--68c53b7909ee2fb48f1979dd.chromatic.com/iframe.html?args=kedroBaseUrl%3Ahttps__COLON____SLASH____SLASH__patcon__DOT__github__DOT__io__SLASH__kedro-polislike-pipelines-san-juan-islands&globals=&id=components-app-kedro-mode--kedro-mode-with-animation&viewMode=story)
  - original polis: [report](https://pol.is/report/r7bhuide6netnbr8fxbyh)

## Background

Polis is a collective intelligence tool for collecting simple agree/disagree data and from that
building maps of the opinion space in which participants reside. This allows sensemaking by
surfacing complexity in the groups that agree/disagree together.

## Goals

- allow for more visibility into existing Polis pipeline
- support exploration of new parameters and algorithms
- support collaboration on these new pipeline variants
- support generation of standardized data types that new UI can be built around
- modularization of pipeline steps
- help determine best architecture for the standalone [`red-dwarf` algorithm library](https://github.com/polis-community/red-dwarf/)

## Usage

The project uses a **branching pipeline** that runs multiple imputer-reducer-clusterer combinations efficiently by sharing intermediate results. This creates a single DAG where preprocessing runs once, and then multiple algorithm combinations branch from the shared preprocessing output.

### Quick Start

```bash
# 1. Run the full pipeline (675 tasks - all combinations)
make run-pipelines PARAMS="polis_url=https://pol.is/report/r3vumzb3w4zccaty6vcan"

# 2. Build the static visualization site
make build-viz POLIS_URL="https://pol.is/report/r3vumzb3w4zccaty6vcan"

# 3. Serve the site locally
make serve-viz
# Visit: http://localhost:8000
```

### Pipeline Execution

#### Full Pipeline
```bash
# Run all combinations (675 tasks, ~2-3 minutes)
make run-pipelines PARAMS="polis_url=https://pol.is/report/r3vumzb3w4zccaty6vcan"
```

#### Tag-Based Filtering
**⚠️ Important**: Tag filtering in Kedro only runs nodes with matching tags. For complex pipelines with dependencies, this can cause execution failures if prerequisite nodes are missing.

```bash
# Run only preprocessing + shared dataset nodes (9 tasks, ~5 seconds)
make run-pipelines TAGS="base" PARAMS="polis_url=https://pol.is/report/r3vumzb3w4zccaty6vcan"

# Run preprocessing + all "mean" imputer combinations (100 tasks, ~30 seconds)
make run-pipelines TAGS="mean" PARAMS="polis_url=https://pol.is/report/r3vumzb3w4zccaty6vcan"
```

**Available Tags:**
- **Stage tags**: `imputer`, `reducer`, `scaler`, `filter`, `clusterer`
- **Component tags**: `mean`, `median`, `zero`, `knn5d`, `pca`, `umap`, `pacmap`, `localmap`, `bestkmeans`, `besthdbscan`
- **Combination tags**: `mean_pca_bestkmeans`, `zero_pacmap_masked_bestkmeans`, etc.
- **Special tags**: `base` (preprocessing + shared), `shared` (dataset generation), `visualization`

### Build & Serve Workflow

#### Building the Static Site
```bash
# Build Kedro Viz static site with data
make build-viz POLIS_URL="https://pol.is/report/r3vumzb3w4zccaty6vcan"

# Alternative: using Polis ID instead of full URL
make build-viz POLIS_ID="r3vumzb3w4zccaty6vcan"
```

#### Serving the Site
```bash
# Serve the built site with CORS support
make serve-viz
# Visit: http://localhost:8000

# The site includes:
# - Interactive Kedro Viz pipeline visualization
# - All generated data files and outputs
# - Scatter plots and projections for each combination
```

### Development Commands

```bash
# Start Kedro Viz development server (live pipeline view)
make dev
# Visit: http://localhost:4141

# Show all available make commands
make help
```

### Performance Comparison

| Command | Tasks | Time | Use Case |
|---------|-------|------|----------|
| Full pipeline | 675 | ~2-3 min | Complete analysis of all combinations |
| `TAGS="base"` | 9 | ~5 sec | Just preprocessing + shared data |
| `TAGS="mean"` | 100 | ~30 sec | All combinations with mean imputation |
| `TAGS="pca"` | ~150 | ~45 sec | All combinations with PCA reduction |

### Troubleshooting

#### Tag Filtering Issues
If you get `DatasetError: Data for MemoryDataset has not been saved yet`, it means the filtered tags don't include all necessary dependencies. Solutions:

1. **Use broader tags**: Instead of specific combinations, use component tags like `mean`, `pca`, `bestkmeans`
2. **Include base preprocessing**: Always include `base` tag when filtering: `TAGS="base,your_tags"`
3. **Run full pipeline first**: Some tag combinations work better after a full pipeline run

#### Build Issues
```bash
# If build fails, reset configuration
make reset-tmp-build-config

# Check if data directory exists
ls -la data/

# Verify Polis URL is accessible
curl -I "https://pol.is/report/r3vumzb3w4zccaty6vcan"
```

#### Common Errors
- **Missing POLIS_URL**: Ensure you provide either `POLIS_URL` or `POLIS_ID` for build commands
- **Port conflicts**: If port 8000 is busy, the serve command will show an error
- **Memory issues**: Large datasets may require more RAM; consider filtering to specific combinations
