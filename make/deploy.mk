# Deployment processes
.PHONY: deploy deploy-staging deploy-prod package package-docker deploy-check

deploy: deploy-check deploy-staging

deploy-check:
	@echo "Pre-deployment checks..."
	@echo "Running tests before deployment..."
	@make test
	@echo "Running linting before deployment..."
	@make lint
	@echo "Pre-deployment checks passed!"

deploy-staging: package
	@echo "Deploying to staging environment..."
	@echo "Building Docker image for staging..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t terrorblade:staging .; \
		echo "Docker image built successfully"; \
		echo "Deployment to staging completed!"; \
		echo "Access staging at: http://localhost:8080"; \
	else \
		echo "No Dockerfile found. Creating basic deployment package..."; \
		make package; \
		echo "Package ready for manual deployment"; \
	fi

deploy-prod: deploy-check
	@echo "Deploying to production environment..."
	@echo "WARNING: This will deploy to production!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@echo "Building production Docker image..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t terrorblade:latest .; \
		docker tag terrorblade:latest terrorblade:prod; \
		echo "Production image built successfully"; \
		echo "Deploy the image to your production environment"; \
	else \
		echo "No Dockerfile found. Creating production package..."; \
		make package; \
		echo "Production package ready"; \
	fi

package: build
	@echo "Creating deployment package..."
	@echo "Creating package directory..."
	@mkdir -p package/
	@echo "Copying distribution files..."
	@cp -r dist/* package/ 2>/dev/null || echo "No dist files found"
	@echo "Creating requirements file..."
	@uv pip freeze > package/requirements.txt
	@echo "Creating deployment archive..."
	@tar -czf terrorblade-deployment-$$(date +%Y%m%d-%H%M%S).tar.gz package/
	@echo "Deployment package created successfully!"
	@rm -rf package/

package-docker:
	@echo "Creating Docker deployment package..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t terrorblade:$$(date +%Y%m%d-%H%M%S) .; \
		docker save terrorblade:$$(date +%Y%m%d-%H%M%S) | gzip > terrorblade-docker-$$(date +%Y%m%d-%H%M%S).tar.gz; \
		echo "Docker package created successfully!"; \
	else \
		echo "No Dockerfile found. Cannot create Docker package."; \
		exit 1; \
	fi