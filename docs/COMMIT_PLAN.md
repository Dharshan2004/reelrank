# Commit phases

This plan groups the entire current working tree, including the earlier correctness work. The four phases below are being committed locally at the user's request; no push is authorized. The commands document the grouping and are not instructions to create duplicate commits. Run any future adaptations from the repository root, one phase at a time. Inspect `git diff --cached` before each commit; use `git restore --staged <path>` if a file contains work you want to keep out.

Use your configured author identity. Do not add a Codex/bot co-author trailer. The existing initial commit is authored by Cursor Agent; these commands do not rewrite it.

## 1. Correct training, evaluation, and serving

Keep the shared feature contract, chronological retrieval, provenance and dependent API changes together: splitting them into cosmetic file-by-file commits would leave incompatible intermediate states. Include the dependencies and lockfile used to run the tests.

```bash
git add reelrank/data/history.py reelrank/data/paths.py \
  reelrank/ranking reelrank/retrieval reelrank/eval reelrank/serve \
  tests/test_pipeline.py tests/test_ranking_features.py \
  tests/test_retrieval_training.py tests/test_runlog.py \
  tests/test_serving_validation.py pyproject.toml uv.lock LICENSE
make test
git diff --cached --check
git diff --cached --stat
# Review the complete staged patch before committing:
git diff --cached
git commit -m "fix: align training, evaluation, and serving contracts"
```

This also includes portable storage directories and an external-storage provenance regression test. MIT is the user-selected code license.

## 2. Preserve safety-label provenance

```bash
git add reelrank/safety/judge.py reelrank/safety/kappa.py \
  tests/test_safety_provenance.py
make test
git diff --cached --check
git diff --cached
git commit -m "fix: version safety labels and preserve human reviews"
```

These changes repair labeling behavior; they do not imply existing labels were regenerated or validated.

## 3. Record evaluated results

```bash
git add results notebooks scripts docs/demo.gif
# Inspect all included files, especially immutable run artifacts:
git diff --cached --stat
git diff --cached --check
git commit -m "docs: record reproducible offline evaluation results"
```

Includes compact source snapshots and per-user evaluation outputs already produced by prior runs. Preserve recorded hashes and historical filenames. The results index distinguishes current measurements, historical measurements, and safety limitations. Do not rerun training merely to change timestamps for this commit.

## 4. Polish setup and project documentation

```bash
git add README.md GLOSSARY.md Makefile Dockerfile .dockerignore \
  configs/default.yaml examples docs
make test example
git diff --cached --check
git diff --cached
git commit -m "docs: simplify setup and organize project documentation"
git status --short
```

`git add configs/default.yaml` stages its deletion. Its reference values now live in `docs/reference-defaults.md`; there was no config loader. Docker mounts generated data/artifacts rather than baking them into the image. This phase includes this plan and the work log.

## Before pushing

Before committing, isolated index snapshots of phase 1 and phase 2 passed 27 and 29 tests respectively, using the existing project environment. This verifies the staged source independently of later unstaged changes; it is not a fresh dependency installation. Phase 3 adds results and presentation artifacts without changing runtime code. The final phase is checked with the tests and synthetic example.

Check `git status --short` for leftovers and `git log -4 --format=full` for messages/authorship. Push only when you intend to publish. CI and contributing guidelines are intentionally omitted for this personal project.
