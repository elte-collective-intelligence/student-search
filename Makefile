.PHONY: build train eval

CONTAINER_TOOL ?= docker
IMAGE ?= ghcr.io/elte-collective-intelligence/student-search:latest
PROJECT_ROOT := $(shell pwd)
LOG_DIR ?= $(PROJECT_ROOT)/search_rescue_logs
BUILD_ARGS ?=
TRAIN_ARGS ?=
EVAL_ARGS ?=

build:
	$(CONTAINER_TOOL) build $(BUILD_ARGS) --label org.opencontainers.image.revision=$(shell git rev-parse HEAD) --label org.opencontainers.image.source=$(IMAGE) --label org.opencontainers.image.created=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ") -t $(IMAGE) -f docker/Dockerfile .

train: build
	@mkdir -p $(LOG_DIR)
	$(CONTAINER_TOOL) run --rm \
		-v $(LOG_DIR):/app/search_rescue_logs \
		$(IMAGE) $(TRAIN_ARGS) ++train.active=true ++eval.active=false ++tensorboard.active=false ++save_folder=search_rescue_logs

eval: build
	@mkdir -p $(LOG_DIR)
	$(CONTAINER_TOOL) run --rm -it \
		-v $(LOG_DIR):/app/search_rescue_logs \
		$(IMAGE) $(EVAL_ARGS) ++train.active=false ++eval.active=true ++tensorboard.active=false ++save_folder=search_rescue_logs

tensorboard: build
	@mkdir -p $(LOG_DIR)
	$(CONTAINER_TOOL) run --rm -it -p 6006:6006 \
		-v $(LOG_DIR):/app/search_rescue_logs \
		$(IMAGE) ++train.active=false ++eval.active=false ++tensorboard.active=true ++save_folder=search_rescue_logs
