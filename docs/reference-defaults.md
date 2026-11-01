# Reference defaults

These values summarize the experiment defaults; this is not an executable configuration file. Use module `--help` and Make variables to change runs. Each result records its effective configuration.

```yaml
seed: 42
data:
  train_end: 20220421
  val_end: 20220428
  history_n: 20
retrieval:
  epochs: 8
  batch_size: 1024
  emb_dim: 64
  temperature: 0.05
  max_eval_users: 2000
  label: long_view
ranking:
  max_train: 400000
  max_val: 80000
  mmoe_experts: 4
  loss_weights: [1.0, 1.5, 1.0]
  selection_split: standard_validation  # fusion weights saved with selected checkpoint
safety:
  n_labels: 1500
  threshold: 0.5
  judge_model: gpt-5.6-luna
serve:
  port: 43123
  retrieve_k: 200
  k: 10
  ranker: lr  # primary standard-validation long_view GAUC winner; MMoE remains available
```
