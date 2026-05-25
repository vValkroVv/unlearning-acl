# DualCF Reproduction Notes

This note keeps operational commands out of the front page. It is docs-only:
it does not replace the production runbook in
[`prod-run-dual-gpu.md`](../prod-run-dual-gpu.md).

## Environment

For lightweight local checks:

```bash
git clone git@github.com:vValkroVv/unlearning-acl.git
cd unlearning-acl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

For the GPU environment used by the runbooks:

```bash
bash setup_vast_env.sh
source .venv/bin/activate
```

For manual installation, `requirements.txt` contains the clean public core
dependencies and `requirements-gpu.txt` lists the GPU/runtime pins used by
`setup_vast_env.sh`. Install CUDA `torch` and `flash-attn` according to your
machine before running full campaigns.

Use placeholders for local data and model mirrors:

```bash
export REPO_ROOT=$PWD
export DATA_ROOT=/path/to/unlearning-data
export ARTIFACT_ROOT=${DATA_ROOT}/artifacts/dualcf
export ALTPO_ARTIFACT_ROOT=${DATA_ROOT}/artifacts/altpo
export OUTPUT_ROOT=${DATA_ROOT}/saves/unlearn
export HF_HOME=${DATA_ROOT}/.hf_home
export HF_DATASETS_CACHE=${DATA_ROOT}/.hf_datasets_cache
export TRITON_CACHE_DIR=${DATA_ROOT}/.triton
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$ARTIFACT_ROOT" "$ALTPO_ARTIFACT_ROOT" "$OUTPUT_ROOT" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"
```

Offline runs normally expect mirrors under names like:

```text
${DATA_ROOT}/SwetieePawsss/DUET
${DATA_ROOT}/SwetieePawsss/exp_r
${DATA_ROOT}/SwetieePawsss/DUET_ft_models
${DATA_ROOT}/models/BASE/Llama-3.1-8B-Instruct
```

## Utility Panel

The production wrapper defaults to Utility-3K. Build it once per machine and
reuse it for endpoint and checkpoint evaluation:

```bash
export UTILITY=3k
export UTILITY_ROOT=${DATA_ROOT}/evals/utility_3k_v1
export BASELINE_CACHE_ROOT=${DATA_ROOT}/saves/eval/utility_baselines
mkdir -p "$UTILITY_ROOT" "$BASELINE_CACHE_ROOT"

python src/tools/build_utility_1k_panel.py \
  --output-dir "$UTILITY_ROOT" \
  --seed 1337 \
  --mmlu-pro 1200 \
  --truthfulqa-bin 600 \
  --arc 600 \
  --winogrande 600 \
  --arc-split test
```

For matched older reruns, set `UTILITY=1k` and use the smaller panel command in
[`prod-run-dual-gpu.md`](../prod-run-dual-gpu.md).

## Artifact Preparation Overview

The reported flow is:

```text
candidate generation -> cleaning -> validation/ranking/repair
-> difficulty scoring -> attribution scoring -> percentile calibration
-> final JSONL artifact
```

Use the production runbook for the exact DUET and RWKU artifact-preparation
commands:

- DUET: [`scripts/duet/prepare_dual_cf_duet_v2.sh`](../scripts/duet/prepare_dual_cf_duet_v2.sh)
- RWKU: [`scripts/rwku/prepare_dual_cf_rwku_v2.sh`](../scripts/rwku/prepare_dual_cf_rwku_v2.sh)
- Full runbook: [`prod-run-dual-gpu.md`](../prod-run-dual-gpu.md)

## Validate Artifacts

DUET:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /path/to/dualcf_rare_v2.jsonl \
  --question-key question \
  --reject-gold-substring \
  --require-short-answer \
  --check-overlap-ratio 0.85 \
  --strict
```

RWKU:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /path/to/dualcf_forget_level2_v2.jsonl \
  --question-key query \
  --reject-gold-substring \
  --require-short-answer \
  --check-overlap-ratio 0.85 \
  --strict
```

The trainer-facing row must include `index`, the prompt field (`question` or
`query`), `answer`, `alternate`, `difficulty_score`, and
`attribution_score`.

## One-Step Smoke Runs

DUET local JSON smoke:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_smoke \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files=/path/to/dualcf_rare_v2.jsonl \
  "cf_dataset_split='train[:2]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

RWKU local JSON smoke:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/rwku/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=rwku_dualcf_smoke \
  forget_split=forget_level2/test \
  retain_split=neighbor_level2/test \
  cf_dataset_path=json \
  cf_dataset_data_files=/path/to/dualcf_forget_level2_v2.jsonl \
  "cf_dataset_split='train[:2]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/rwku_dualcf_smoke
```

Expected smoke checks:

- training finishes without OOM,
- `dualcf_*` routing logs appear,
- `difficulty_score` and `attribution_score` reach the trainer,
- an adapter checkpoint is saved.

## Reported Campaign Wrapper

The current paper-facing full method is the GeneralCF-style configuration with
NPO-SAM, span masking, full difficulty/attribution routing, and no reported
rarity route:

```bash
source .venv/bin/activate

export ARTIFACT_ROOT=/path/to/artifacts/dualcf
export ALTPO_ARTIFACT_ROOT=/path/to/artifacts/altpo
export OUTPUT_ROOT=/path/to/saves/unlearn
export HF_HOME=/path/to/.hf_home
export HF_DATASETS_CACHE=/path/to/.hf_datasets_cache
export TRITON_CACHE_DIR=/path/to/.triton
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPU_ID=0
SEEDS="42 179 1137" \
METHOD_VARIANTS="general_cf" \
ADDITIONAL_LOSS=NPO-SAM \
ROUTING=full \
SPAN_ADDITIONAL=true \
SPAN_CF_BRANCH=true \
DISABLE_RARITY_ROUTES=true \
DISABLE_DIFFICULTY_ROUTES=false \
DISABLE_ATTRIBUTION_ROUTES=false \
RARITY_NEG_GAINS="0.0" \
RARITY_CF_GAINS="0.0" \
SPAN_MODE=lcs \
SPAN_ALT_SHARED_TOKEN_WEIGHT=0.0 \
SPAN_ALT_UNIQUE_TOKEN_WEIGHT=1.0 \
SPAN_ORIG_SHARED_TOKEN_WEIGHT=0.0 \
SPAN_ORIG_UNIQUE_TOKEN_WEIGHT=1.0 \
BETAS=0.1 \
GAMMAS=1.0 \
SPAN_SAM_RHO=0.01 \
SPAN_SAM_ADAPTIVE=false \
bash scripts/dualcf/run_campaign_one_lr.sh "${GPU_ID}" 1e-4 all
```

The wrapper also supports `METHOD_VARIANTS=full`, but the explicit
`general_cf` block above documents the paper-facing decomposition and avoids
implicit defaults.

For the broad method family, prefer the grouped examples in
[`prod-run-dual-gpu.md`](../prod-run-dual-gpu.md) instead of pasting a large
command block here.

## Post-Run Analysis

```bash
python scripts/calc_cos_sim.py --path_to_saves "$OUTPUT_ROOT" --gpu 0
python scripts/calc_wrong_generations.py --path_to_saves "$OUTPUT_ROOT"
bash package_saves.sh \
  --path_to_saves "$OUTPUT_ROOT" \
  --out_path /path/to/saves-clean \
  --save_eval 0
```

Use [`src/tools/build_structured_saves.py`](../src/tools/build_structured_saves.py)
and [`src/tools/build_results_combine_tables.py`](../src/tools/build_results_combine_tables.py)
for structured table extraction after packaged saves are available.
