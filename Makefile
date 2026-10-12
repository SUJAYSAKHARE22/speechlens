.PHONY: install test lint corpus eval demo serve docker clean

install:
	pip install -e ".[dev,app]"

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests

corpus:            ## 3 simulated speakers + datasets (needs espeak-ng)
	speechlens make-dev-corpus --out data/dev_corpus

eval: corpus       ## regenerate the evaluation reports (about 8 min on one core)
	speechlens evaluate --out docs/evaluation
	speechlens evaluate --oracle --out docs/evaluation
	speechlens evaluate --no-calibrate --name dev_synthetic_uncalibrated --out docs/evaluation
	python scripts/make_eval_summary.py

demo: corpus
	speechlens make-demo --corpus data/dev_corpus --out data/demo

serve:
	speechlens serve --data data/demo

docker:
	docker build -t speechlens . && docker run --rm -p 8000:8000 speechlens

clean:
	rm -rf data/dev_corpus .pytest_cache .ruff_cache
