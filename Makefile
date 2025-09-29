# Kedro Polis Pipelines Makefile
# Based on the original Makefile structure with targets for build, run-pipelines, and dev

run-pipelines: ## Run the pipeline script (use PIPELINES= and PARAMS= env vars)
	@echo "🚀 Running Kedro pipelines..."
	@if [ -n "$(PIPELINES)" ]; then \
		if [ -n "$(PARAMS)" ]; then \
			echo "Running pipelines: $(PIPELINES) with params: $(PARAMS)"; \
			python scripts/run_pipelines.py $(PIPELINES) --params "$(PARAMS)"; \
		else \
			echo "Running pipelines: $(PIPELINES)"; \
			python scripts/run_pipelines.py $(PIPELINES); \
		fi; \
	else \
		if [ -n "$(PARAMS)" ]; then \
			echo "Running all pipelines with params: $(PARAMS)"; \
			python scripts/run_pipelines.py --all --params "$(PARAMS)"; \
		else \
			echo "Running all pipelines"; \
			python scripts/run_pipelines.py --all; \
		fi; \
	fi

dev: ## Run kedro viz for development
	@echo "🔧 Starting Kedro Viz development server..."
	kedro viz

set-build-polis-id: ## Set build_polis_id in base globals (requires POLIS_ID env var)
	@echo "🔧 Setting build_polis_id to $(POLIS_ID) in base globals..."
	@if [ -z "$(POLIS_ID)" ]; then \
		echo "❌ Error: POLIS_ID environment variable is required"; \
		echo "Usage: POLIS_ID=6carwc4nzj make set-build-polis-id"; \
		exit 1; \
	fi
	@sed -i.bak 's/build_polis_id: null/build_polis_id: $(POLIS_ID)/' conf/base/globals.yml
	@echo "✅ build_polis_id set to $(POLIS_ID) in conf/base/globals.yml"
	@echo "⚠️  Remember to run 'make reset-build-polis-id' after building to reset to null"

reset-build-polis-id: ## Reset build_polis_id to null in base globals
	@echo "🔄 Resetting build_polis_id to null in base globals..."
	@sed -i.bak 's/build_polis_id: [^[:space:]]*/build_polis_id: null/' conf/base/globals.yml
	@rm -f conf/base/globals.yml.bak
	@echo "✅ build_polis_id reset to null in conf/base/globals.yml"

build: ## Build static site in build directory (requires POLIS_ID env var)
	@echo "🏗️  Building static site..."
	@if [ -z "$(POLIS_ID)" ]; then \
		echo "❌ Error: POLIS_ID environment variable is required"; \
		echo "Usage: POLIS_ID=6carwc4nzj make build"; \
		exit 1; \
	fi
	@mkdir -p build
	@echo "🔧 Temporarily setting build_polis_id to $(POLIS_ID)..."
	@sed -i.bak 's/build_polis_id: null/build_polis_id: $(POLIS_ID)/' conf/base/globals.yml
	@echo "🔧 Building Kedro Viz with polis_id=$(POLIS_ID)..."
	@kedro viz build --include-previews || (echo "Build failed, restoring globals..."; mv conf/base/globals.yml.bak conf/base/globals.yml; exit 1)
	@echo "🔄 Restoring build_polis_id to null..."
	@mv conf/base/globals.yml.bak conf/base/globals.yml
	python scripts/copy_data.py
	@echo "🔧 Fixing API file paths..."
	python scripts/fix_api_paths.py
	@echo "✅ Build completed! Static site ready in build/ directory"

serve: ## Serve the build directory with Python HTTP server (with CORS headers)
	@echo "🌐 Starting HTTP server for build directory with CORS support..."
	@if [ ! -d "build" ]; then \
		echo "❌ Build directory not found. Run 'make build' first."; \
		exit 1; \
	fi
	python scripts/serve_with_cors.py

.PHONY: help build run-pipelines dev serve

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
