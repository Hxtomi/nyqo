# Nyqo n1 — development priorities

This page tracks priorities for improving and distributing Nyqo n1.

## Evaluation

Complete the reproducibility record for the model comparisons reported in the [README](../README.md). Record exact checkpoint IDs, revisions, inference settings, and evaluation outputs for each additional model.

Additional training seeds and targeted ablations would help establish how much the current result depends on initialization, target selection, and the retention objective. The existing [evaluation record](results.md) describes what is supported by the current measurements.

## Training

The current results show incomplete retention of previously successful answers and no demonstrated improvement on IFEval. Further training should address these weaknesses while tracking performance across all six evaluated domains.

Potential changes include broader training-data coverage, improved target quality, and stronger retention methods. Their value needs to be established by comparison with n1 and the base model.

## Hugging Face release

Publish Nyqo n1 as a Hugging Face model repository containing the weights, configuration, tokenizer, chat template, and model card. GitHub remains the source for code, training details, and evaluation documentation.

The release target is [Hxtomi/nyqo](https://huggingface.co/Hxtomi/nyqo), which currently remains private. The model card is maintained in [release/huggingface/README.md](../release/huggingface/README.md), alongside the model license, modification notice, and file manifest. See the [release instructions](../release/README.md) for staging and verification.

### Preparation

- Upload the six verified checkpoint files to the private release repository, then validate the model-card example and CLI on a fresh GPU system. Record the tested Hub revision and observed output before opening access.
- Switch the default downloader from the private recovery archive to the tested release revision and update the usage instructions.
- Include the applicable model license and upstream attribution with the weights; keep source and data terms documented separately.
- Measure inference memory use and latency on specified hardware.
- Document which training and evaluation artifacts can be obtained and reproduced.

The [model card](../MODEL_CARD.md), [usage guide](usage.md), and [artifact inventory](../artifacts/README.md) describe the current model and available materials.
