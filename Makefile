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

set-tmp-build-config: ## Set temporary build config in base globals (requires POLIS_ID or POLIS_URL env var)
	@echo "🔧 Setting temporary build configuration in base globals..."
	@if [ -n "$(POLIS_URL)" ]; then \
		echo "Using POLIS_URL: $(POLIS_URL)"; \
		sed -i.bak 's|build_polis_url: null|build_polis_url: "$(POLIS_URL)"|' conf/base/globals.yml; \
		echo "✅ build_polis_url set to $(POLIS_URL) in conf/base/globals.yml"; \
	elif [ -n "$(POLIS_ID)" ]; then \
		echo "Using POLIS_ID: $(POLIS_ID)"; \
		sed -i.bak 's/build_polis_id: null/build_polis_id: $(POLIS_ID)/' conf/base/globals.yml; \
		echo "✅ build_polis_id set to $(POLIS_ID) in conf/base/globals.yml"; \
	else \
		echo "❌ Error: Either POLIS_ID or POLIS_URL environment variable is required"; \
		echo "Usage: POLIS_ID=6carwc4nzj make set-tmp-build-config"; \
		echo "   or: POLIS_URL=https://polis.example.com/6carwc4nzj make set-tmp-build-config"; \
		exit 1; \
	fi
	@echo "⚠️  Remember to run 'make reset-tmp-build-config' after building to reset to null"

reset-tmp-build-config: ## Reset temporary build config to null in base globals
	@echo "🔄 Resetting temporary build configuration to null in base globals..."
	@sed -i.bak 's|build_polis_id: [^[:space:]]*|build_polis_id: null|' conf/base/globals.yml
	@sed -i.bak 's|build_polis_url: "[^"]*"|build_polis_url: null|' conf/base/globals.yml
	@rm -f conf/base/globals.yml.bak
	@echo "✅ Temporary build configuration reset to null in conf/base/globals.yml"

build: ## Build static site in build directory (requires POLIS_ID or POLIS_URL env var)
	@echo "🏗️  Building static site..."
	@mkdir -p build
	@$(MAKE) set-tmp-build-config
	@echo "🔧 Building Kedro Viz..."
	@kedro viz build --include-previews || (echo "Build failed, restoring globals..."; $(MAKE) reset-tmp-build-config; exit 1)
	@echo "🔄 Restoring globals to null..."
	@$(MAKE) reset-tmp-build-config
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

# Legacy targets for backward compatibility
set-build-polis-id: set-tmp-build-config # Legacy alias for set-tmp-build-config
reset-build-polis-id: reset-tmp-build-config # Legacy alias for reset-tmp-build-config
set-build-polis-url: set-tmp-build-config # Legacy alias for set-tmp-build-config
reset-build-polis-url: reset-tmp-build-config # Legacy alias for reset-tmp-build-config

.PHONY: help build run-pipelines dev serve

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
