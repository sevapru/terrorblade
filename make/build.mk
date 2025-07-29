# Build processes
.PHONY: build build-clean build-install build-wheel build-sdist

build: build-clean
	@echo "Building project..."
	@echo "Creating source distribution and wheel..."
	python3 -m build
	@echo "Build completed successfully!"
	@echo "Artifacts created in dist/:"
	@ls -la dist/ 2>/dev/null || echo "No artifacts found"

build-clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Build artifacts cleaned!"

build-install: build-clean
	@echo "Installing dependencies for build..."
	uv pip install build wheel twine
	@echo "Build dependencies installed!"

build-wheel: build-clean build-install
	@echo "Building wheel distribution..."
	python3 -m build --wheel
	@echo "Wheel built successfully!"

build-sdist: build-clean build-install
	@echo "Building source distribution..."
	python3 -m build --sdist
	@echo "Source distribution built successfully!"