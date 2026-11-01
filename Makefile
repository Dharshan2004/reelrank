# Prefer the project venv (uv sync); fall back to python3 on PATH.
PYTHON ?= $(or $(wildcard .venv/bin/python),python3)
PORT ?= 43123
# Torch + FAISS both ship OpenMP on macOS; without this, train/eval can SIGSEGV after epochs.
export KMP_DUPLICATE_LIB_OK ?= TRUE
export OMP_NUM_THREADS ?= 1

USERS ?= 8
SPAWN_RATE ?= 4
DURATION ?= 30s
EVAL_USERS ?= 2000
RANKER ?= lr

.PHONY: install data features baselines train-retrieval train-ranker train-safety safety-kappa eval serve loadtest loadtest-httpx demo-gif test all eval-retrieval

install:
	uv sync --extra dev --locked

data:
	$(PYTHON) -m reelrank.data.download
	$(PYTHON) -m reelrank.data.split
	$(PYTHON) -m reelrank.data.features

features:
	$(PYTHON) -m reelrank.data.features

baselines:
	$(PYTHON) -m reelrank.retrieval.baselines

train-retrieval:
	$(PYTHON) -m reelrank.retrieval.train --epochs 8

eval-retrieval:
	$(PYTHON) -m reelrank.retrieval.eval_artifacts

train-ranker:
	$(PYTHON) -m reelrank.ranking.train

train-safety:
	$(PYTHON) -m reelrank.safety.judge
	$(PYTHON) -m reelrank.safety.classifier
	$(PYTHON) -m reelrank.safety.cascade

safety-kappa:
	$(PYTHON) -m reelrank.safety.kappa

eval:
	$(PYTHON) -m reelrank.eval.pipeline_eval --ranker $(RANKER) --max-users $(EVAL_USERS)
	$(PYTHON) -m reelrank.eval.make_readme_table

serve:
	REELRANK_RANKER=$(RANKER) $(PYTHON) -m uvicorn reelrank.serve.app:app --host 0.0.0.0 --port $(PORT)

loadtest:
	$(PYTHON) -m locust -f reelrank/serve/locustfile.py --headless \
		-u $(USERS) -r $(SPAWN_RATE) -t $(DURATION) \
		--host http://127.0.0.1:$(PORT)

loadtest-httpx:
	$(PYTHON) -m reelrank.serve.loadtest --base-url http://127.0.0.1:$(PORT)

demo-gif:
	$(PYTHON) scripts/make_demo_gif.py --base-url http://127.0.0.1:$(PORT)

test:
	$(PYTHON) -m pytest -q

all: data baselines train-retrieval train-ranker train-safety eval

.PHONY: example
example:
	$(PYTHON) -m examples.synthetic
