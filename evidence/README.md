# Evidence

These records support **Nyqo n1** (`nyqo-n1`). Historical labels inside the archived records identify internal stages of development. The original records and identifiers remain unchanged.

| File | Meaning |
|:--|:--|
| [release_validation.json](release_validation.json) | Checkpoint upload, file verification, and a local CPU generation check; fresh GPU validation pending |
| [comparison_summary.json](comparison_summary.json) | Current comparison counts and the author's internal evaluation scoring decisions of 27 October 2026 |
| [closeout.json](closeout.json) | Aggregate N2 acceptance, external transfer, overlap audit, and preservation |
| [final_receipt.json](final_receipt.json) | Safe aggregate closed-final result |
| [finalist_selection.json](finalist_selection.json) | Selected model identity and DEV decision fixed before external/final evaluation |
| [train_freeze_v2.json](train_freeze_v2.json) | Corrected data and order hashes at preparation time |
| [provenance.json](provenance.json) | Exact origin of preserved source/receipts and local integrity bindings |

Preparation receipts preserve their original historical statuses. The later closeout describes completion. The copied closeout adds the current private research repository name after its rename; its original source blob is recorded in provenance.

These are evidence records, not a complete evaluation dataset. Closed final questions, answers, rubrics, and individual grades are absent. See [Results](../docs/results.md) and [Artifacts](../artifacts/README.md) for interpretation and availability.

The comparison summary records the finalized scores and supersedes provisional score reporting in the archived receipts. It records the author's scoring decisions, not new inference runs or independent regrading.
