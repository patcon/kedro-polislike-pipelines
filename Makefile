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

build: ## Build static site in build directory
	@echo "🏗️  Building static site..."
	@mkdir -p build
	python scripts/copy_data.py
	@echo "🔧 Fixing API file paths..."
	python scripts/fix_api_paths.py
	@echo "✅ Build completed! Static site ready in build/ directory"

serve: ## Serve the build directory with Python HTTP server
	@echo "🌐 Starting HTTP server for build directory..."
	@if [ ! -d "build" ]; then \
		echo "❌ Build directory not found. Run 'make build' first."; \
		exit 1; \
	fi
	@echo "🚀 Server starting at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	cd build && python -m http.server 8000

.PHONY: help build run-pipelines dev serve

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
