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

```bash
# Build static site
uv run make build
# or: make build

# Run all pipelines
uv run make run-pipelines
# or: make run-pipelines

# Run specific pipelines with parameters
uv run make run-pipelines PIPELINES=bestkmeans PARAMS="polis_id=r29kkytnipymd3exbynkd"
# or: make run-pipelines PIPELINES=bestkmeans PARAMS="polis_id=r29kkytnipymd3exbynkd"

# Start development server
uv run make dev
# or: make dev

# Serve build directory
uv run make serve
# or: make serve

# Show help
make
# or: make help
```
