.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/02_gbm_monotone_challenger.py -o notebooks/02_gbm_monotone_challenger.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1500 notebooks/02_gbm_monotone_challenger.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info
