# Makefile for Cosmos Explorer

# Variables
PROJECT_NAME = cosmos-explorer
PYTHON = python3
PIP = pip3

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the application locally"
	@echo "  deploy      - Deploy to Google App Engine"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up temporary files"
	@echo "  help        - Show this help message"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Run the application locally
.PHONY: run
run:
	$(PYTHON) app.py

# Deploy to Google App Engine
.PHONY: deploy
deploy:
	gcloud app deploy

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest

# Clean up temporary files
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache