# Nyqo n1 — evaluation record

Results for **Nyqo n1** (`nyqo-n1`) refer to the weight hash in [the checkpoint manifest](../artifacts/nyqo-n1.json). Archived N2 and N2-FINAL204 labels refer to the internal development phase and its final evaluation bank, not a public Nyqo n2 model.

A score on a custom subset is not a full-benchmark leaderboard score. Sources for the n1/base evaluation are the [closeout receipt](../evidence/closeout.json), [final receipt](../evidence/final_receipt.json), and [selection receipt](../evidence/finalist_selection.json).

The finalized comparison appears in the [README](../README.md#results) and [release model card](../release/huggingface/README.md#evaluation). Nyqo n1 scores **56/204** on FINAL. The project author confirmed this final total on 27 October 2026. Paired intervals have not been recomputed following the final scoring decision.

## Current model comparison

The table below records counts of tasks passed from the reported comparison. The SmolLM2, Qwen2.5-0.5B, Granite-350M, and Qwen 4B rows are descriptive references reported by the author. Exact checkpoint IDs, revisions, and per-model inference records for these additional models are not included in the published evidence, so their scores cannot be independently reproduced from this release.

| Model | FINAL / 204 | ARC-Easy / 594 | IFEval / 210 | MBPP+ / 100 |
|:--|--:|--:|--:|--:|
| LFM2.5-350M (base) | 45 | 336 | 159 | 3 |
| Nyqo n1 | 56 | 370 | 153 | 13 |
| SmolLM2 | 43 | 290 | 70 | 22 |
| Qwen2.5-0.5B | 46 | 339 | 52 | 37 |
| Granite-350M | 45 | 353 | 105 | 31 |
| Qwen 4B (reference) | 138 | 450 | 181 | 63 |

On 27 October 2026, the project author, Artur Markov-Tsoy, finalized four FINAL scores: Nyqo n1 **56/204 (27.45%)**, Qwen2.5-0.5B **46/204 (22.55%)**, Granite-350M **45/204 (22.06%)**, and Qwen 4B **138/204 (67.65%)**. These replace the earlier reported bounds in the current comparison. They are the author's scoring decisions, not new model runs or independently recomputed item grades. The other reported scores are unchanged.

The [comparison record](../evidence/comparison_summary.json) stores the finalized counts and scoring decisions. The earlier paired analyses remain historical records.

The Qwen 4B column is a larger-model reference. n1's KL sources are the base model and an earlier Nyqo checkpoint, as described in [the training objective](method.md#objective).

## Frozen public transfer subsets

| Metric | Tasks | B0 correct | Nyqo correct | B0, % | Nyqo, % | Change, pp |
|:--|--:|--:|--:|--:|--:|--:|
| ARC-Easy acc_norm | 594 | 336 | 370 | 56.57 | 62.29 | +5.72 |
| IFEval strict prompt | 210 | 159 | 153 | 75.71 | 72.86 | −2.86 |
| MBPP+ subset, base + additional tests | 100 | 3 | 13 | 3.00 | 13.00 | +10.00 |

ARC-Easy and IFEval are the two primary follow-up endpoints. The MBPP+ subset (100 tasks) is a separate secondary diagnostic. Historical artifact filenames and receipt keys identify this same slice as `CODE100`; those identifiers are retained to preserve the evidence bindings.

| Paired family-cluster interval for score change | 95% interval, pp | 97.5% interval, pp |
|:--|:--|:--|
| ARC-Easy | +3.03 to +8.59 | +2.53 to +8.92 |
| IFEval | −7.18 to +1.44 | Not reproduced in the supplied aggregate receipt |
| MBPP+ subset (100 tasks), secondary | +3.92 to +16.38 | Not a primary endpoint |

The follow-up uses 10,000 bootstrap replicates. The 97.5% individual primary intervals implement a Bonferroni adjustment for two comparisons; the ARC interval remains positive. These intervals describe sampled task distributions, not variability across training seeds.

**ARC-Easy.** Zero-shot causal likelihood scores each original answer choice after a question/answer prefix. The primary score divides summed choice log likelihood by the original answer's length in characters. This is answer selection, not generated-answer grading.

**IFEval.** The pinned official checker receives the original response without cleanup. Strict prompt accuracy requires all checked instructions to pass. Both models receive a 2,048-token generation limit in a 4,096-token total window. Ten B0 answers hit the limit, and six of those pass the official checker; Nyqo n1 has no cutoffs. A separate completion-required sensitivity gives 153/210 for both. The official 159/210 versus 153/210 remains the primary result. Neither improvement nor equivalence is established.

**MBPP+ subset (100 tasks).** One greedy response may contain raw Python or one Python Markdown block. A frozen extractor performs no repair, import synthesis, or resampling. A success must pass base and additional EvalPlus tests. Extraction failures and cutoffs remain in the denominator. There are 87 family clusters. All 13 Nyqo n1 successes also obey the allowed format; base-only counts are B0 3 and Nyqo n1 16. Format compliance rises from 45/100 to 97/100, so some of the change concerns the output interface. Absolute programming accuracy is still low.

These follow-up subsets and protocols were frozen before their own responses, after limitations of earlier diagnostics were observed. They were not a plan fixed before every earlier experiment.

## Internal final bank

N2-FINAL204 has 34 tasks in each of six domains. FULL is a binary complete-task-and-rubric success metric. Semantic grading uses a blinded GPT-6.1 Sol judge per domain at medium reasoning effort, with objective checks where applicable. The separate evaluator role is not an independent human replication.

| Domain, 34 tasks each | B0 | Nyqo n1 | Required n1 minimum |
|:--|--:|--:|--:|
| Source use | 8 | 7 | 7 |
| Instructions | 5 | 7 | 4 |
| Reasoning / mathematics | 0 | 2 | — |
| Knowledge / uncertainty | 12 | 19 | 10 |
| Dialogue context | 17 | 18 | 13 |
| Code | 3 | 3 | 3 |
| **Total** | **45** | **56** | **52** |

The expanded-model acceptance rule is at least 15% relative macro gain, at most 500M total parameters, and strictly less than 25% relative regression in every domain with nonzero B0 accuracy. Equal domain sizes make macro accuracy equal to total accuracy here. Zero-baseline domains remain in the average without an invented minimum.

The relative gain of Nyqo n1 is **24.44%**, or **5.39 percentage points**, meeting the acceptance rule. Three actual B0 runs score 45/204 with identical final text; run 1 is the reference. These repeats check deterministic execution, not independent training variation. K500 scores 53/204 on this same final bank.

Nyqo n1 retains 26 of B0's 45 successful items, loses 19, and gains 30. Source use drops by 12.5%. The earlier 5,000-replicate family-cluster bootstrap has a 95% macro-difference envelope of **−1.33 to +11.59 pp**, including zero; it has not been recomputed following the final scoring decision. Passing the project rule does not establish a population-wide positive effect or preservation of every previous success.

Closed final prompts, answers, rubrics, and individual grades are not distributed in this repository. Only the safe aggregate receipts are included.

## Development selection

DEV84 has 14 tasks per domain. All candidate identities below were fixed before final evaluation.

| Model | FULL / 84 | Code / 14 | Meets every domain limit |
|:--|--:|--:|:--|
| B0 | 18 | 3 | Baseline |
| K500 | 22 | 2 | No |
| R100 | 19 | 2 | No |
| R200 | 25 | 1 | No |
| **Nyqo n1, selected** | **19** | **3** | **Yes** |
| C200 | 22 | 2 | No |

Nyqo n1 is the sole new candidate satisfying the domain limits. Its 5.56% relative aggregate DEV gain is below the final target; the plan permits the subsequent borderline final check. R200's higher total does not compensate for its code regression.

## Earlier diagnostics remain separate

| Original 64-task profile | B0 | K500 | Nyqo n1 |
|:--|--:|--:|--:|
| IFEval strict, 768-token response limit | 52/64 | 47/64 | 46/64 |
| MBPP+, original no-stripping output contract | 0/64 | 0/64 | 0/64 |

Every original MBPP+ response begins with a Markdown fence and fails raw Python parsing. Those zeros remain the result under that contract. The later 100-task MBPP+ subset permits fence extraction and is a different diagnostic, not a replacement score. The IFEval generation windows also differ and cannot be pooled.

## Overlap coverage and reproducibility

All 1,600 training inputs are covered by input-only comparisons. The last 150 inputs were checked after outcomes, against public ARC 2,376, IFEval 541, and MBPP+ 378 pools. No exact matches were found in that added coverage. Fifty-five retrieved semantic candidate pairs were reviewed, with no additional concrete task/family conflicts or unresolved candidate pairs. Frozen samples and scores were unchanged.

This bounded retrieval plus review does not guarantee detection of every paraphrase. Training targets were not compared in that audit; pretraining overlap is unknown.

The exact public slices, raw responses, checker/extractor implementations, and sample manifests are preserved in private transfer supplements. This repository publishes the aggregate evidence and implementation snapshot, but does not claim a complete public rerun of the custom slices. See [artifact availability](../artifacts/README.md). The additional models in the latest comparison summary are not covered by the paired analyses in these archived receipts.
