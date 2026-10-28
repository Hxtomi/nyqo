# Earlier architecture exploration

Before the experiment that produced the released Nyqo N1 checkpoint, we tested both unchanged-size and expanded variants. These were exploratory runs on an earlier 4,000-example corpus and a different, repeatedly used 204-task evaluation bank. They are separate from the final corrected 1,600-example stage and its current internal evaluation results.

| Variant | Total parameters | Trainable parameters | Full / 204 | Code / 34 | Base successes retained / 55 |
|:--|--:|--:|--:|--:|--:|
| LFM2.5-350M (base) | 354,483,968 | 0 | 55 | 9 | 55 |
| Unchanged-size A | 354,483,968 | 287,375,104 | 64 | 0 | 26 |
| Unchanged-size B | 354,483,968 | 287,375,104 | 56 | 1 | 24 |
| Expanded E497 | 497,119,488 | 142,635,520 | 69 | 3 | 34 |
| Expanded E497K | 497,119,488 | 142,635,520 | 72 | 4 | 39 |

Counts use the archived 23 October review. Both unchanged-size variants increased the aggregate score but failed the mathematics and code retention limits; B also missed the aggregate-gain requirement. The expanded variants achieved higher aggregate scores and retained more baseline successes, but still failed the code retention limit. All four trained variants failed the earlier acceptance criteria.

All four runs record 500 updates, a learning rate of 1e-5, the same original base revision and batch order. A/B trained existing blocks; the expanded variants trained inserted blocks while freezing the originals, and E497K used retention distillation. Target files and objectives varied across recipes, so these are recipe comparisons rather than a controlled test of model capacity alone.

This history motivated retaining the expanded backbone for the subsequent corrected-data experiment. It does not show that unchanged-size fine-tuning is generally ineffective or that extra parameters alone caused the released checkpoint's improvement. A matched unchanged-size control on the final 1,600-example stage remains absent. Historical scores are not part of the current benchmark comparison.

The [machine-readable summary](../evidence/architecture_history.json) records counts, scope and source identities. The underlying archive is private; closed evaluation questions, model answers and individual grades are not included here.

[Current training method](method.md) · [Current evaluation results](results.md)
