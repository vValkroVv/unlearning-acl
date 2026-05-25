# DualCF Results

All values are percentages. `F↓` is the original-answer forget score on forget
prompts, so lower is better. `H↑` is holdout/locality on neighboring or matched
non-forget examples. `U↑` is aggregate general utility over MMLU-Pro,
TruthfulQA-Binary, WinoGrande, and ARC-Challenge.

The metrics must be read together. Lowering `F` by collapsing `H` or sharply
reducing `U` is destructive forgetting, not selective unlearning.

Numbers here are copied from the paper table sources in
[`docs/paper_tables`](paper_tables). The active comparison excludes IdkDPO and
uses the 19-baseline no-IdkDPO scope at learning rate `1e-4`, epoch 5, over
seeds 42, 179, and 1137 where seed variation is available.

## Final DualCF Summary

| Split | DualCF F↓ | DualCF H↑ | DualCF U↑ |
|---|---:|---:|---:|
| DUET Rare | 0.7 (0.1) | 96.5 (0.2) | 56.9 (0.2) |
| DUET Popular | 6.5 (1.7) | 98.8 (0.6) | 57.2 (0.3) |
| DUET Merged | 1.9 (0.4) | 98.7 (0.5) | 56.6 (0.5) |
| RWKU | 8.8 (0.1) | 96.8 (0.5) | 57.3 (0.3) |

## Main Comparison Matrix

Each cell is `F / H / U` with sample standard deviation in parentheses.

| Method | Rare F/H/U | Popular F/H/U | Merged F/H/U | RWKU F/H/U |
|---|---:|---:|---:|---:|
| DualCF | 0.7 (0.1) / 96.5 (0.2) / 56.9 (0.2) | 6.5 (1.7) / 98.8 (0.6) / 57.2 (0.3) | 1.9 (0.4) / 98.7 (0.5) / 56.6 (0.5) | 8.8 (0.1) / 96.8 (0.5) / 57.3 (0.3) |
| DPO | 58.2 (7.2) / 99.8 (0.2) / 56.2 (0.6) | 18.1 (4.9) / 99.7 (0.4) / 56.9 (0.2) | 33.7 (5.0) / 100.0 (0.0) / 56.1 (0.3) | 12.8 (1.2) / 95.8 (0.3) / 57.1 (0.9) |
| GA | 0.0 (0.0) / 0.0 (0.0) / 36.6 (5.3) | 0.0 (0.0) / 0.0 (0.0) / 54.8 (0.3) | 0.0 (0.0) / 0.1 (0.1) / 45.6 (2.4) | 0.1 (0.0) / 0.0 (0.0) / 31.8 (4.8) |
| NPO | 67.2 (5.1) / 99.2 (0.2) / 57.0 (0.2) | 85.9 (3.1) / 99.4 (0.5) / 57.2 (0.0) | 73.5 (1.6) / 99.5 (0.2) / 57.1 (0.3) | 55.3 (3.7) / 95.9 (0.1) / 57.7 (0.6) |
| NPO-SAM | 13.8 (9.4) / 96.0 (1.9) / 57.1 (0.2) | 83.6 (4.9) / 98.6 (0.4) / 57.5 (0.2) | 36.5 (16.5) / 97.8 (1.1) / 57.2 (0.6) | 30.3 (9.3) / 94.7 (0.2) / 56.6 (0.3) |
| LoKU | 0.8 (0.1) / 97.3 (0.7) / 47.0 (2.1) | 5.5 (1.2) / 97.7 (0.4) / 49.6 (1.1) | 4.8 (0.3) / 99.4 (0.6) / 43.6 (2.1) | 1.9 (0.4) / 97.4 (0.2) / 45.8 (0.9) |
| SimNPO | 33.1 (10.6) / 99.1 (0.2) / 56.9 (0.1) | 15.5 (2.4) / 99.6 (0.4) / 56.9 (0.1) | 34.0 (6.9) / 99.9 (0.1) / 56.9 (0.1) | 57.3 (1.9) / 98.7 (0.2) / 57.6 (0.9) |
| Adaptive-RMU | 96.2 (0.2) / 96.5 (0.3) / 57.4 (0.1) | 85.5 (8.4) / 96.5 (0.1) / 57.5 (0.2) | 93.3 (0.6) / 96.3 (0.5) / 57.6 (0.2) | 16.2 (3.0) / 81.9 (0.6) / 57.5 (0.1) |
| FLAT | 0.1 (0.0) / 0.1 (0.0) / 52.0 (4.0) | 0.0 (0.0) / 0.1 (0.0) / 53.0 (4.1) | 0.1 (0.0) / 0.1 (0.0) / 46.3 (3.2) | 0.4 (0.0) / 0.3 (0.0) / 44.9 (9.6) |
| Unilogit | 1.7 (0.8) / 94.7 (0.5) / 56.1 (0.5) | 7.4 (4.8) / 98.7 (0.4) / 57.3 (0.2) | 3.2 (1.0) / 98.5 (1.5) / 56.6 (0.5) | 2.5 (1.9) / 95.9 (0.3) / 58.2 (1.2) |
| STAT | 31.7 (27.0) / 99.5 (0.6) / 56.1 (0.6) | 13.1 (4.0) / 99.2 (0.2) / 56.5 (0.4) | 7.8 (2.5) / 99.7 (0.2) / 56.7 (0.4) | 7.7 (3.4) / 98.0 (0.2) / 56.6 (0.8) |
| SatImp | 19.3 (16.8) / 99.6 (0.3) / 56.3 (0.2) | 9.1 (5.5) / 99.6 (0.3) / 57.2 (0.1) | 10.8 (4.9) / 99.5 (0.5) / 56.5 (0.4) | 41.7 (3.4) / 98.5 (0.2) / 57.4 (0.1) |
| UnDIAL | 89.4 (0.2) / 96.4 (0.4) / 57.5 (0.1) | 90.3 (0.1) / 96.5 (0.2) / 57.5 (0.1) | 90.1 (0.2) / 96.7 (0.3) / 57.5 (0.2) | 56.7 (0.4) / 81.4 (0.4) / 58.2 (0.3) |
| RMU | 59.8 (5.4) / 93.3 (0.2) / 55.8 (0.3) | 77.5 (6.8) / 96.1 (0.2) / 57.5 (0.4) | 71.3 (2.6) / 95.0 (0.5) / 53.5 (0.8) | 10.6 (3.4) / 80.7 (0.3) / 56.6 (0.9) |
| WGA | 1.6 (0.2) / 97.3 (1.4) / 56.5 (0.6) | 7.9 (2.7) / 99.3 (0.2) / 56.8 (0.3) | 7.8 (3.6) / 99.0 (0.8) / 56.4 (0.3) | 6.2 (1.5) / 97.4 (0.2) / 58.4 (0.5) |
| AltPO | 36.7 (4.6) / 99.9 (0.2) / 56.3 (0.3) | 3.4 (3.0) / 100.0 (0.0) / 55.6 (0.8) | 18.1 (4.6) / 100.0 (0.0) / 56.7 (0.1) | 20.0 (3.2) / 98.8 (0.8) / 55.2 (0.3) |
| TPO | 61.7 (12.9) / 99.4 (0.3) / 56.4 (0.2) | 9.4 (0.6) / 99.1 (0.6) / 56.5 (0.3) | 39.7 (2.5) / 99.7 (0.1) / 56.5 (0.5) | 34.1 (8.4) / 98.6 (0.2) / 57.4 (0.4) |
| GradDiff | 0.4 (0.1) / 81.7 (1.4) / 56.2 (0.3) | 3.1 (0.8) / 85.9 (3.2) / 57.0 (0.7) | 2.1 (0.4) / 86.6 (0.2) / 56.5 (0.1) | 1.9 (0.1) / 75.3 (1.9) / 55.5 (1.8) |
| CE-U | 0.0 (0.0) / 0.0 (0.0) / 51.3 (1.7) | 0.4 (0.6) / 1.7 (3.0) / 45.4 (1.8) | 0.0 (0.0) / 0.0 (0.1) / 46.5 (2.5) | 0.0 (0.0) / 0.0 (0.0) / 43.9 (9.3) |
| PDU | 1.8 (1.4) / 92.0 (2.6) / 56.6 (0.6) | 59.9 (13.3) / 96.0 (0.7) / 57.2 (0.4) | 36.5 (3.0) / 97.4 (0.5) / 56.1 (0.5) | 9.8 (1.7) / 94.1 (0.3) / 57.3 (0.7) |

## Artifact Quality

Same full objective, only selected counterfactual target changes.

| Split | Artifact | F↓ | H↑ | U↑ |
|---|---|---:|---:|---:|
| DUET Rare | CF-Single | 32.8 (0.9) | 98.5 (0.2) | 56.8 (0.3) |
| DUET Rare | CF-Multi | 5.6 (1.6) | 97.7 (0.9) | 56.0 (0.0) |
| DUET Rare | CF-Repair | 0.7 (0.1) | 96.5 (0.2) | 56.9 (0.2) |
| DUET Popular | CF-Single | 2.9 (0.6) | 98.2 (0.2) | 56.9 (0.7) |
| DUET Popular | CF-Multi | 6.5 (1.7) | 98.8 (0.6) | 57.2 (0.3) |
| DUET Popular | CF-Repair | 6.5 (1.7) | 98.8 (0.6) | 57.2 (0.3) |
| DUET Merged | CF-Single | 17.6 (1.1) | 99.0 (1.0) | 56.2 (0.5) |
| DUET Merged | CF-Multi | 5.9 (2.7) | 99.2 (0.5) | 56.5 (0.3) |
| DUET Merged | CF-Repair | 1.9 (0.4) | 98.7 (0.5) | 56.6 (0.5) |
| RWKU | CF-Single | 12.6 (0.4) | 96.3 (0.3) | 56.6 (0.8) |
| RWKU | CF-Multi | 8.8 (0.1) | 96.8 (0.5) | 57.3 (0.3) |
| RWKU | CF-Repair | 8.8 (0.1) | 96.8 (0.5) | 57.3 (0.3) |

## Ablation Sources

The full ablation matrices are kept as source tables:

- [`03_ablation_single_short.tex`](paper_tables/03_ablation_single_short.tex):
  fixed `CF-Single` artifact.
- [`04_ablation_multi_short.tex`](paper_tables/04_ablation_multi_short.tex):
  fixed `CF-Multi` artifact.
- [`05_ablation_repair_short.tex`](paper_tables/05_ablation_repair_short.tex):
  fixed `CF-Repair` artifact.

The key pattern is artifact-conditioned: under `CF-Multi`, full DualCF lowers
DUET Rare `F` from SimpleCE's `22.9` to `5.6`; after `CF-Repair`, the gap
shrinks to `1.5` versus `0.7`. This supports the paper's interpretation that
target construction and routed optimization are complementary.

## Interpretation

- Counterfactual target construction is a core part of factual unlearning. On
  DUET Rare, `CF-Single -> CF-Multi -> CF-Repair` changes `F` from
  `32.8 -> 5.6 -> 0.7` under the same optimizer.
- Raw deletion is not enough. GA, FLAT, and CE-U can drive `F` near zero, but
  frequently collapse holdout/locality or utility.
- The comparison is split-matched. DUET Rare, DUET Popular, DUET Merged, and
  RWKU artifacts/runs are separate objects.
