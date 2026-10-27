# Preserved scientific implementation

These files are copied from the C100 recovery source snapshot, commit `9420ff6a88ae82e71c8938e672b893e19e4d7d9d`, in the private research repository now named `Hxtomi/nyqo-devs`.

| File | Purpose |
|:--|:--|
| [n1/expanded_runtime.py](n1/expanded_runtime.py) | Native identity expansion, frozen backbone, expanded loader |
| [n1/runtime.py](n1/runtime.py) | Shared tokenizer, parser, and historical N1 training utilities |
| [n2/runtime/n2_runtime.py](n2/runtime/n2_runtime.py) | Corrected mixed CE/native-KL objective, precision, masking, restore |
| [n2/runtime/train.py](n2/runtime/train.py) | Scientific training driver |
| [n2/runtime/run_scientific_arm_v2.py](n2/runtime/run_scientific_arm_v2.py) | Fixed-order R/C run entry point |
| [n2/inference/generate.py](n2/inference/generate.py) | Original internal generation profile and receipts |

They retain their original absolute cloud paths, software assumptions, hashes, and GPU constraints. They do not run directly from this subdirectory. The N1 files supply expansion/shared utilities; the N2 runtime and driver define the selected checkpoint's training objective.

[provenance.json](../../evidence/provenance.json) binds each source to its original Git blob and SHA256. The small portable interface in [nyqo/](../../nyqo/) is new packaging and does not replace this implementation record.

Exact retraining additionally needs TRAIN1600_v2, its saved order, B0, and the K500 reference used by native-target KL. Those dependencies and training state are privately archived; see [Artifacts](../../artifacts/README.md). No optimizer-state pickle is included or automatically loaded by the inference interface.
