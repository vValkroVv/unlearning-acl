# DualCF Artifacts

DualCF treats the replacement answer as part of the method. Artifact
construction chooses the positive target before training; routing and
optimization consume the frozen target later.

## Artifact Row

A final trainer-facing row has the following fields:

```json
{
  "index": 17,
  "question": "...",
  "answer": "...",
  "alternate": "...",
  "difficulty_score": 0.73,
  "attribution_score": 0.18
}
```

RWKU uses `query` instead of `question`. Sidecars may store additional
provenance, candidate pools, and compatibility features. `rarity_score` may be
present in some rows for compatibility; the reported optimizer ignores it.

## Levels

| Artifact | Meaning |
|---|---|
| `CF-Single` | One generated/selected alternate per forget row. |
| `CF-Multi` | Eight generated candidates per row from OpenAI/Codex/ChatGPT-Pro sidecars; validation and deterministic ranking select one final alternate. |
| `CF-Repair` | A repair pass over `CF-Multi` where repair is needed. It is active on DUET Rare and the rare side of DUET Merged, and equal to `CF-Multi` on DUET Popular and RWKU. |

All final reported artifacts preserve the full forget-example count:

| Split | Rows |
|---|---:|
| DUET Rare | 482 |
| DUET Popular | 482 |
| DUET Merged | 964 |
| RWKU Level 2 | 2879 |

## Construction Boundary

Artifact construction:

```text
forget row -> candidate pool -> hard validation -> ranking -> repair/fallback -> alternate
```

Routing and optimization:

```text
alternate + difficulty_score + attribution_score -> loss weights during training
```

The boundary is strict. `alternate` is selected by the artifact pipeline before
training. Difficulty and attribution scores do not choose, rank, validate, or
repair candidate answers; they only set loss weights in the trainer.

## Validation Rules

Final artifacts should be validated before training:

- no empty `alternate`,
- no exact answer copy,
- no original-answer substring leakage when `--reject-gold-substring` is used,
- short-answer format for factual QA targets,
- overlap ratio below the configured threshold,
- full row preservation for the target split,
- calibrated `difficulty_score` and `attribution_score` fields.

DUET validation:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /path/to/dualcf_rare_v2.jsonl \
  --question-key question \
  --reject-gold-substring \
  --require-short-answer \
  --check-overlap-ratio 0.85 \
  --strict
```

RWKU validation:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /path/to/dualcf_forget_level2_v2.jsonl \
  --question-key query \
  --reject-gold-substring \
  --require-short-answer \
  --check-overlap-ratio 0.85 \
  --strict
```

## Repair Activity By Split

| Split | Repair behavior |
|---|---|
| DUET Rare | Repair is active over rare rows; the reported repaired artifact changes brittle selected targets and leaves unresolved rows at zero. |
| DUET Popular | Repair is inactive; `CF-Repair = CF-Multi`. |
| DUET Merged | Repair is active only on the rare side; popular-side rows are unchanged. |
| RWKU Level 2 | Repair is inactive; `CF-Repair = CF-Multi`. |

The source tools are
[`make_counterfactuals.py`](../src/tools/make_counterfactuals.py),
[`clean_counterfactuals.py`](../src/tools/clean_counterfactuals.py),
[`score_difficulty.py`](../src/tools/score_difficulty.py),
[`score_attribution.py`](../src/tools/score_attribution.py),
[`calibrate_dual_cf_scores.py`](../src/tools/calibrate_dual_cf_scores.py),
and [`validate_dual_cf_artifact.py`](../src/tools/validate_dual_cf_artifact.py).
