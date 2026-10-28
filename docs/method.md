# Nyqo n1 — training method

This document describes how **Nyqo n1** was trained: architecture changes, answer verification, the training objective, and checkpoint selection. It also records the data exposure and compute used in development.

## Backbone and expansion

The starting point, B0, is the instruction-tuned `LiquidAI/LFM2.5-350M` at revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`: 354,483,968 parameters and 16 native layers.

After original layers 3, 7, 11, and 15, the implementation inserts independent copies of original layers 14 and 15. The actual residual output projections in each inserted operator and feed-forward block are zero-initialized. This makes the added blocks identity mappings at initialization. Original weights remain frozen, but the resulting network's behavior can change as the inserted blocks are trained.

| Quantity | Value |
|:--|--:|
| Expanded layers | 24 |
| Inserted layer positions, zero-based | 4, 5, 10, 11, 16, 17, 22, 23 |
| Frozen original parameters | 354,483,968 |
| Trainable inserted parameters | 142,635,520 |
| Total parameters | 497,119,488 |

The [original expansion implementation](../research/original/n1/expanded_runtime.py) is preserved with source hashes in [provenance](../evidence/provenance.json). This uses established identity expansion; the project does not propose a new expansion algorithm.

## Why use an expanded backbone?

Earlier unchanged-size variants increased the aggregate score but failed the required balance across mathematics and code. Related expanded variants had higher aggregate and baseline-retention counts, although they still failed the earlier code limit. These observations motivated retaining expansion for the next corrected-data experiment. The [earlier architecture exploration](architecture-history.md) used a different corpus and evaluation bank; it is not a matched ablation of the final training stage.

## Data and targets

The corrected frozen stage contains 1,600 English examples: 800 code tasks and 160 tasks in each of source use, instructions, reasoning/math, knowledge/uncertainty, and dialogue context.

Targets follow a fixed priority:

1. Keep an unchanged, verified K500 answer when it is correct.
2. Otherwise keep an unchanged, verified B0 answer when it is correct.
3. Otherwise use a checked corrected or external reference.

Corrected targets are not classified as unchanged native answers. Code targets are checked against task interfaces, constraints, examples, and executable behavior; the other domains use the corresponding semantic or calculation checks.

Two earlier code targets were corrected for their literal floating-point semantics before the valid R/C comparison. The earlier R_v1 run is excluded from selection and retained in total cost accounting. The remaining 1,598 rows, IDs, task inputs, and order are unchanged. The [freeze receipt](../evidence/train_freeze_v2.json) records this preparation stage; its historical “not executed” status is superseded by the later [closeout receipt](../evidence/closeout.json).

## Objective

Each effective group contains four code and four other examples. The loss is the sum of three unit-weight terms:

- CE averaged over the group's active code target tokens.
- CE averaged over the group's active other-domain target tokens.
- Full-vocabulary KL from the appropriate frozen source model, averaged over the group's unchanged native target tokens.

The KL term is zero in groups without native targets. Corrected/external targets receive CE only. Prompt and padding positions are masked; targets include EOS. Code and other-domain CE are normalized separately, rather than pooled by token count.

The full stage contains 214,502 active response/EOS tokens, of which 22,682 are native target tokens. Native code answers occur in only 30 of the 800 code examples. Retention coverage is therefore limited even when the conditional KL weight is one.

## Optimization and selection

| Setting | Recorded value |
|:--|:--|
| Hardware | One NVIDIA RTX 3060, 12 GB |
| Software | PyTorch 2.11.0+cu128; Transformers 5.8.1 |
| Precision during training | FP32 inserted parameters; BF16 frozen parameters and autocast |
| Optimizer | AdamW; LR 1e-5; betas 0.9/0.95; epsilon 1e-8 |
| Regularization | Weight decay 0.01; gradient norm clipping at 1 |
| Batch | Microbatch 1; effective group 8 |
| Schedule | 6-update warmup; cosine horizon 200 updates; final factor 0.1 |
| Seed | 20261024 |
| Training context | At most 2,048 tokens; no silent target truncation |

Arm C starts from clean B0 plus identity expansion. Arm R starts from the previously adapted K500 checkpoint. Both valid arms use fresh optimizers, the same corrected data and fixed order, and 200 additional updates. Their lifetime training histories differ, so this is a practical initialization comparison, not an equal-compute isolation of expansion.

Nyqo n1 is arm C's midpoint. It is fixed on DEV before final or external responses are generated. It is the only new candidate meeting all domain retention limits on DEV, although its aggregate DEV gain is below the 15% goal. The existing plan permits a borderline final check. No candidate is reselected after final evaluation.

## Exposure and cost

| Scope | Updates | Distinct examples seen | Active target/EOS tokens |
|:--|--:|--:|--:|
| Selected Nyqo n1 checkpoint | 100 | 800 | 107,728 |
| Each complete valid R/C arm | 200 | 1,600 | 214,502 |
| All scientific attempts, including excluded R_v1 | 600 | Not an additional unique-data count | 643,862 |

Five technical test updates are recorded separately. The selected checkpoint uses the first half of a 200-update schedule, not an independently scheduled 100-update run.

The author reports US$19.23 in cloud instance rental. This is not the cost of pretraining LFM2.5, all research subscriptions, or all project work. Active target tokens exclude prompt tokens and do not measure total training/evaluation FLOPs.

## What this experiment can establish

The fixed checkpoint meets the project's internal acceptance rule and improves two specific public diagnostics. The available controls do not isolate the contributions of expansion, target selection, or KL. An unchanged-size SFT/LoRA control matched to the final 1,600-example stage, a no-KL ablation, additional seeds, and measured inference efficiency are absent.

[Results](results.md) describes the evidence and its limits. [Artifact availability](../artifacts/README.md) distinguishes preserved files from publicly downloadable ones.
