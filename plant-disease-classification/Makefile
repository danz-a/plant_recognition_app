.PHONY: install data baselines cnn tune transfer evaluate test lint clean

install:          ## Install the package and dev tooling
	pip install -e ".[tuning,explain,stats,dev]"

data:             ## Download PlantVillage and build the canonical split
	python scripts/prepare_data.py

baselines:        ## Dummy, logistic regression and random forest floors
	python scripts/train_baselines.py

cnn:              ## Train the from-scratch baseline CNN
	python scripts/train_cnn.py

tune:             ## Bayesian hyper-parameter search
	python scripts/tune_cnn.py

transfer:         ## Fine-tune DenseNet-121
	python scripts/train_transfer.py --backbone densenet121

evaluate:         ## Score the baseline on validation and the sealed test set
	python scripts/evaluate.py --run cnn_baseline

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
