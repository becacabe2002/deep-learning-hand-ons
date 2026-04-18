# Makefile for Deep Learning Hands-On

.PHONY: help train clean

help:
	@echo "Deep Learning Hands-On Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make train DIR=<module_path>    Run training for a specific module"
	@echo ""
	@echo "Example:"
	@echo "  make train DIR=numpy_from_scratch.01_linear_nn"
	@echo "  make train DIR=frameworks_modern/01_cnn_foundations"

train:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR variable is required."; \
		echo "Usage: make train DIR=numpy_from_scratch.01_linear_nn"; \
		exit 1; \
	fi
	$(eval MODULE_PATH := $(subst .,/,$(DIR)))
	@if [ ! -f "$(MODULE_PATH)/train.py" ]; then \
		echo "Error: $(MODULE_PATH)/train.py not found."; \
		exit 1; \
	fi
	python $(MODULE_PATH)/train.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
