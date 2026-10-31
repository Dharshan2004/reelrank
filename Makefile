PYTHON ?= python
PORT ?= 43123

.PHONY: install data features baselines train-retrieval train-ranker train-safety eval serve loadtest test all

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m reelrank.data.download
	$(PYTHON) -m reelrank.data.split
	$(PYTHON) -m reelrank.data.features

features:
	$(PYTHON) -m reelrank.data.features

baselines:
	$(PYTHON) -m reelrank.retrieval.baselines

train-retrieval:
	$(PYTHON) -m reelrank.retrieval.train

train-ranker:
	$(PYTHON) -m reelrank.ranking.train

train-safety:
	$(PYTHON) -m reelrank.safety.judge
	$(PYTHON) -m reelrank.safety.classifier
	$(PYTHON) -m reelrank.safety.cascade

eval:
	$(PYTHON) -m reelrank.eval.pipeline_eval
	$(PYTHON) -m reelrank.eval.make_readme_table

serve:
	$(PYTHON) -m uvicorn reelrank.serve.app:app --host 0.0.0.0 --port $(PORT)

loadtest:
	$(PYTHON) -m reelrank.serve.loadtest --base-url http://127.0.0.1:$(PORT)

test:
	$(PYTHON) -m pytest -q

all: data baselines train-retrieval train-ranker train-safety eval
