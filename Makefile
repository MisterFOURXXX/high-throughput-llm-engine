VENV      := venv
PYTHON    := $(shell if [ -x "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; else echo "python3"; fi)
PIP       := $(shell if [ -x "$(VENV)/bin/pip" ];    then echo "$(VENV)/bin/pip";    else echo "pip3";    fi)
TF_DIR    := configs

.PHONY: all help clean-all setup run-all simulation test clean clean-terraform nuke

all: help

help:
	@echo "Targets:"
	@echo "  clean-all        Clean repo (venv, caches, TF state)"
	@echo "  setup            Full environment setup"
	@echo "  simulation       Run NCCL vs XDP vs Gloo simulation"
	@echo "  test             Run pytest"
	@echo "  run-all          Run full workflow"
	@echo "  clean            Remove build artifacts"
	@echo "  clean-terraform  Remove Terraform state"
	@echo "  nuke             Full cleanup (venv + state + artifacts)"

clean-all:
	@bash scripts/clean_all.sh

setup:
	@bash scripts/setup.sh

simulation:
	@$(PYTHON) scripts/run_simulation.py \
		--nodes 2 --gpus-per-node 4 --iterations 50 \
		--output-dir artifacts

test:
	@$(PYTHON) -m pytest tests/ -v

run-all:
	@bash scripts/run_all.sh

clean:
	@rm -rf benchmarks/build artifacts/*
	@touch artifacts/.gitkeep
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-terraform:
	@rm -rf $(TF_DIR)/.terraform $(TF_DIR)/.terraform.lock.hcl \
	        $(TF_DIR)/terraform.tfstate $(TF_DIR)/terraform.tfstate.backup \
	        $(TF_DIR)/tfplan

nuke: clean clean-terraform
	@rm -rf $(VENV) .pytest_cache .coverage htmlcov