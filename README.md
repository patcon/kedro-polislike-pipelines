# Kedro Pipeline: Polis Classic

This project is an attempt to use Kedro to model [Polis](https://pol.is/home)-like data pipelines.

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
