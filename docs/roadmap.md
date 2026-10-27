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

The checkpoint repository is [Hxtomi/nyqo](https://huggingface.co/Hxtomi/nyqo). The model card is maintained in [release/huggingface/README.md](../release/huggingface/README.md), alongside the model license, modification notice, and file manifest. See the [release instructions](../release/README.md) for staging and verification.

### Preparation

- Completed: upload the six checkpoint files and release manifest, and verify their sizes and hashes.
- Completed: switch the default downloader to the pinned Hxtomi/nyqo revision and update the usage instructions.
- Completed: include the model license and upstream attribution. Local CPU loading and a short generation check also passed.
- Pending: validate the model-card example and CLI on a fresh GPU system. Record the tested Hub revision and observed output before declaring the CUDA entry points verified.
- Measure inference memory use and latency on specified hardware.
- Document which training and evaluation artifacts can be obtained and reproduced.

The [model card](../MODEL_CARD.md), [usage guide](usage.md), and [artifact inventory](../artifacts/README.md) describe the current model and available materials.
