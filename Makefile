.PHONY: build train eval

CONTAINER_TOOL ?= docker
IMAGE ?= student-search:latest
PROJECT_ROOT := $(shell pwd)
LOG_DIR ?= $(PROJECT_ROOT)/search_rescue_logs
BUILD_ARGS ?=
TRAIN_ARGS ?=
EVAL_ARGS ?=

build:
	$(CONTAINER_TOOL) build $(BUILD_ARGS) -t $(IMAGE) -f docker/Dockerfile .

train: build
	@mkdir -p $(LOG_DIR)
	$(CONTAINER_TOOL) run --rm \
		-v $(LOG_DIR):/app/search_rescue_logs \
		$(IMAGE) $(TRAIN_ARGS) ++train.active=true

eval: build
	@mkdir -p $(LOG_DIR)
	$(CONTAINER_TOOL) run --rm -it \
		-v $(LOG_DIR):/app/search_rescue_logs \
		$(IMAGE) $(EVAL_ARGS) ++eval.active=true
