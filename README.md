# ReelRank

A reproducible short-video recommendation prototype on **KuaiRand-Pure**: two-tower retrieval → caption safety filter → ranking → top-k API.

The project compares simple and neural models, tests training/serving consistency, and records both successful and unsuccessful experiments. **LR is the default serving ranker**, chosen by long-view GAUC on standard validation; DeepFM and MMoE remain comparison models. All rankers receive the same train-fitted features. LR has additive effects, so user features change probabilities but do not alone change its item ordering; two-tower retrieval provides personalized candidates. DeepFM/MMoE can learn user–item interactions.

```mermaid
flowchart LR
  req[User request] --> tower[Two-tower user embedding]
  tower --> flat[Exact FAISS top-200]
  flat --> safety[Caption rules + classifier]
  safety --> rank[LR long-view ranking]
  rank --> feed[Top-10 + scores + timings]
  experiment[MMoE experiment] -. optional ranker .-> rank
```


## Quickstart

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run from the repository root:

```bash
uv sync --extra dev --locked
make test
make example
```

The example fits a tiny ranker on synthetic interactions, filters a candidate list, and prints recommendations using the same scoring code as the API. It downloads no data, writes no model artifacts, and makes no quality claim.

## Results

| Stage | Model / measurement | Result |
|---|---|---|
| Retrieval | Popularity | Recall@200 0.05842 |
| Retrieval | ItemCF | Recall@200 0.05350 |
| Retrieval | Two-tower, five training seeds | Recall@200 **0.06666 ± 0.00204** (sample SD) |
| Retrieval comparison | Seed-mean two-tower minus popularity | +14.1% relative; paired absolute-difference 95% CI [0.00092, 0.01576] |
| Retrieval comparison | Seed-mean two-tower minus itemcf | +24.6% relative; paired absolute-difference 95% CI [0.00549, 0.02076] |
| Ranking | LR long-view GAUC | 0.64845 validation / 0.59627 random-exposure |
| Ranking | DEEPFM long-view GAUC | 0.64390 validation / 0.59893 random-exposure |
| Ranking | MMOE long-view GAUC | 0.61726 validation / 0.58518 random-exposure |
| Serving | Original seed-42 bundle; 8 clients, 60 seconds, macOS ARM | 48.36 QPS; p50 31 ms / p99 100 ms; 0 failures / 2,890 requests |
| FAISS | FLAT, original index benchmark | p50 0.082 ms / p99 0.394 ms |
| FAISS | HNSW, original index benchmark | p50 0.056 ms / p99 0.171 ms |

Retrieval uses the same 2,000-user cohort for all seeds and both baselines. The intervals bootstrap **seed-averaged per-user recall differences**, conditional on these five fitted models; they do not cover all training uncertainty. All five point estimates exceed both baselines, but three of five per-seed intervals versus popularity cross zero (all five versus ItemCF exclude zero). Per-seed intervals and results are in the [sweep summary](results/retrieval_seed_sweep/SUMMARY.md). The current serving bundle remains seed 42 (Recall@200 0.06956); no best-seed selection was performed.

The random log sparsely observes outcomes in a 7,583-item candidate pool. Unobserved user–item outcomes are unknown; Recall@200 measures recovery of observed long-view positives, not complete user relevance. A uniformly sampled set of 200 distinct candidates has theoretical expected recall `200 / 7,583 ≈ 0.0264`; this is an analytical reference, not a measured run. Compare models under the same evaluation contract, without treating sparsity as an explanation for every shortfall.

**End-to-end uncertainty remains:** the original LR cascade with safety scored NDCG@10 0.003587 versus 0.002790 for popularity→LR (+28.5% point estimate), but the paired absolute-difference interval [-0.000265, +0.001892] crosses zero. The retrieval sweep does not establish a cascade or online improvement.

LR remains the serving default because it had the highest standard-validation long-view GAUC. Exact FAISS remains the default at this small pool size. The ANN timings above came from a local run during concurrent training, not an isolated performance study; HNSW recall versus Flat was 0.99182. Serving QPS includes think time and is achieved throughput, not capacity.

Sources: [sweep JSON](results/retrieval_seed_sweep.json), [complete results index](results/README.md), [generated table](results/RESULTS_TABLE.md), and [evaluation limitations](docs/evaluation.md). The existing [analysis notebook](notebooks/analysis.ipynb) covers the original single-seed evaluation; the sweep summary is the multi-seed record.

## Reproduce the experiment

After installing dependencies:

```bash
make data                       # download KuaiRand, create splits and features
make baselines                  # popularity and ItemCF
make train-retrieval             # two-tower model and FAISS index
make train-ranker                # LR, DeepFM, MMoE
make train-safety                # see optional paid labeling below
make eval                       # LR, safety on/off, paired intervals
make eval RANKER=mmoe
make serve                      # port 43123; LR and exact FAISS
```

In another terminal:

```bash
curl 'http://127.0.0.1:43123/recommend?user_id=1&k=10'
make loadtest
```

Full reproduction requires downloading data and training models; it is separate from the quickstart. Data and artifacts are not included in Git. Python/pip users can instead create and activate a virtual environment and run `python -m pip install -e '.[dev]'`; this does not use the uv lockfile.

`make train-safety` uses paid LLM labeling if `OPENAI_API_KEY` is set; otherwise it falls back to rules. To explicitly use the no-key path, run `env -u OPENAI_API_KEY make train-safety`. This produces different labels from the recorded LLM-source experiment. Serving never calls an LLM. See the [evaluation and safety limitations](docs/evaluation.md).

To serve MMoE, use `make serve RANKER=mmoe`. Old checkpoints without the saved feature transformer require retraining. See [reference defaults](docs/reference-defaults.md) and each module's `--help`; settings come from CLI arguments and Make variables.

## Repository map

| Path | Purpose |
|---|---|
| `reelrank/data/` | Downloading, splits, features, chronological histories |
| `reelrank/retrieval/` | Two-tower training, baselines, FAISS search |
| `reelrank/ranking/` | Shared feature contract, LR/DeepFM/MMoE, scoring |
| `reelrank/safety/` | Caption rules, labeling, classifier, filtering |
| `reelrank/eval/` | Metrics, cascade evaluation, immutable run records |
| `reelrank/serve/` | FastAPI application and load tests |
| `tests/`, `examples/` | Regression tests and a synthetic walkthrough |
| `results/`, `notebooks/` | [Results index](results/README.md) and analysis |
| `docs/`, `scripts/` | Methodology, work log, notebook/demo generation |

## Docker and local storage

Build the application image after installing Docker. Generate data and artifacts with the commands above before starting the service:

```bash
docker build -t reelrank .
docker run --rm -p 43123:43123 \
  --mount "type=bind,src=$(pwd)/artifacts,dst=/app/artifacts,readonly" \
  --mount "type=bind,src=$(pwd)/data/processed,dst=/app/data/processed,readonly" \
  reelrank
```

The image excludes datasets, model artifacts, and experiment archives. `REELRANK_DATA_DIR`, `REELRANK_ARTIFACTS_DIR`, and `REELRANK_RESULTS_DIR` override storage directories; set them before starting Python. Source checkouts default to repository-local directories; installed packages default to the current working directory.

## Further reading

- [Evaluation protocol and limitations](docs/evaluation.md)
- [Seed-sweep protocol and reproduction](docs/seed-sweep.md)
- [Results and provenance](results/README.md)
- [Implementation work log](docs/WORK_LOG.md)
- [Interview discussion notes](docs/interview-notes.md)
- [KuaiRand dataset documentation](https://kuairand.com/)

Code is released under the [MIT license](LICENSE). Dataset and third-party model licenses remain separate.
