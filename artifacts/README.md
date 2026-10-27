# Artifact inventory

Updated 28 October 2026. “Preserved” describes archived evidence; it does not imply anonymous public access.

This inventory describes the **Nyqo n1** (`nyqo-n1`) checkpoint and its supporting materials. Historical archive paths, hashes, and receipt identifiers are preserved.

| Artifact | Location | Availability |
|:--|:--|:--|
| Expansion, training, original generation source | [research/original/](../research/original/) | Included |
| Safe results and source bindings | [evidence/](../evidence/) | Included |
| Nyqo n1 weight, config, tokenizer, template | [nyqo-n1.json](nyqo-n1.json) | Uploaded to Hxtomi/nyqo; pinned download supported |
| Corrected TRAIN1600_v2 and fixed order | Same model recovery payload | Privately preserved |
| K500 reference and B0 dependency | Recovery payload / original upstream | Preserved; not copied into this Git repository |
| Nyqo n1 optimizer, scheduler, RNG, exposure | Recovery payload, `training_state.pt` | Privately preserved |
| Exact ARC/IFEval transfer slices and responses | `supplements/transfer_final_v1` | Privately preserved |
| MBPP+ subset (100 tasks), checker/extractor, input150 audit | Independent closeout supplement | Privately preserved |
| Closed internal evaluation questions and individual responses/grades | Original evaluator workspace, outside this release | Not found in checked shared backups; current raw availability unverified |

## Immutable locations

- [Nyqo n1 recovery snapshot](https://huggingface.co/Exequted/Nyqo/tree/1e59bda5f5209ab22952f7fd31d5350dba243008/n2/C100_v2/recovery-20261025T173555Z)
- [Independent closeout supplement](https://huggingface.co/Exequted/Nyqo/tree/f5dec7b04f34e16db276f06cfa7ceec121f8d10e/n2/C100_v2/recovery-20261025T173555Z/supplements/independent-closeout-20261025-yuna-v1)
- [Original private source recovery commit](https://github.com/Hxtomi/nyqo-devs/tree/9420ff6a88ae82e71c8938e672b893e19e4d7d9d/Recovery/N2/C100-20261025T173555Z)

The repository was renamed from `nyqo` to `nyqo-devs` when the clean presentation repository was prepared. Historical receipts can retain the former repository name.

The closed final bank's safe aggregate receipts are included, but its raw prompts, responses, and individual grades were not recovered in the shared backups checked on 26 October. The finalized [comparison](../docs/results.md#current-model-comparison) records **56/204** for n1. The missing item records limit independent regrading. The historical closeout records evaluator separation; it is not proof of a currently accessible raw backup.

The supplement containing the 100-task MBPP+ subset has 42 members; the closeout records a repeat download with all member SHA256 checks passing. Historical filenames use the identifier `CODE100` for this slice. The code archive here is checked independently by [verify_repository.py](../scripts/verify_repository.py).

## Important hashes

| Artifact | SHA256 |
|:--|:--|
| Nyqo n1 weights | `6fcd99004b78850b5a323224a9a9833c9bcb13faa8eb762092aed8922fcb0078` |
| TRAIN1600_v2 | `1dca9ef3ae6cc16e3c69037dfc8810b117c007c5711886bf9b3489f950cc8ef0` |
| Fixed order file | `a15a1a58687690c843e6b0386fb690881fb0554059771a8f54103ec0887af2f3` |
| Nyqo n1 training state | `a167ac9cd094d8323816b92628a2f06a6e4627cc34bd89bbf940831e7670df94` |

The public checkpoint manifest is an inference-file allowlist. It deliberately omits the optimizer pickle and training corpus. It must not be used to claim that the closed final bank or a complete public rerun package is included.
