# ReelRank

An end-to-end short-video recommender prototype built on KuaiRand, with standard-validation selection and exploratory evaluation on random-exposure logs.

## Language

**Candidate pool**:
The fixed set of videos that may be recommended (~7.5k in KuaiRand-Pure).
_Avoid_: catalogue (unless meaning coverage over this pool), corpus

**Retrieval**:
The first stage: from the candidate pool, select a shortlist of likely videos for a user (here: top-200).
_Avoid_: recall stage (ambiguous with the Recall@K metric), search

**Safety filter**:
A stage after retrieval that removes or down-ranks captions that violate a content policy (rules, classifier, optional LLM band).
_Avoid_: moderation (broader product process), censor

**Ranking**:
The stage that scores filtered candidates for engagement and orders the final top-k feed. LR long-view scoring is the serving default; MMoE predicts click / long_view / like for a multi-task experiment.
_Avoid_: sorting, re-ranking (unless a later second ranker is added), weighted sum (that is fusion only)

**Fusion**:
A weighted combination of multi-task ranking probabilities into one score for ordering the feed.
_Avoid_: ranking (the whole stage), attention (not the mechanism here)

**Cascade**:
The ordered serving pipeline: retrieval → safety filter → ranking → top-k.
_Avoid_: pipeline (too generic), funnel

**Random-exposure log**:
A historical log where the platform showed videos to users at random. It helps assess exposure-policy bias but still has missing user-item outcomes. Current use is exploratory: older project versions used it for model selection.
_Avoid_: online experiment, A/B test, test log, training set

**Standard log**:
The usual production-style impression log (biased by the previous policy); used for training and secondary splits, not the main claim metric.
_Avoid_: feed log, biased log (ok in speech once defined)

**Engagement label**:
A binary user feedback signal on an impression — in this project: `is_click`, `long_view`, `is_like` (plus `is_hate` only as a safety sanity check).
_Avoid_: reward, CTR (too narrow), interaction (too vague), interaction parameter

**Lift vs popularity**:
The relative change of the full cascade’s primary metric over a popularity→same-ranker baseline on the same cohort and safety policy. Report absolute values and paired uncertainty alongside lift; a positive point estimate alone does not establish a reliable improvement.
_Avoid_: accuracy, absolute NDCG (as the headline claim)
