# Nyqo n1 Hugging Face release preparation

Target: [Hxtomi/nyqo](https://huggingface.co/Hxtomi/nyqo). The staging helper below is intended for private preparation uploads.

The prepared model page is [huggingface/README.md](huggingface/README.md). It accompanies the original [LICENSE](huggingface/LICENSE), a [modification notice](huggingface/NOTICE), the [comparison figure](huggingface/assets/nyqo-benchmarks.svg), [navigation badges](huggingface/assets/badges/), and a [file manifest](huggingface/manifest.json).

The six checkpoint files are not checked into Git. Their expected bytes and hashes identify the selected n1 checkpoint. The weights SHA256 is `6fcd99004b78850b5a323224a9a9833c9bcb13faa8eb762092aed8922fcb0078`.

## Stage the checkpoint

Use an existing checkpoint directory containing the six files:

```bash
python release/prepare_hf.py --source-dir /path/to/checkpoint
```

Without `--source-dir`, the helper downloads only those files from the pinned Hxtomi/nyqo checkpoint revision in [artifacts/nyqo-n1.json](../artifacts/nyqo-n1.json). Standard Hugging Face authentication is used for source and destination access. An optional separate source credential can be supplied through `NYQO_SOURCE_TOKEN`; the legacy `NYQO_ARCHIVE_TOKEN` name is also accepted.

Run this from a workspace on the approved work drive. Downloads and the Hub cache are placed beside the release bundle.

## Verify and upload privately

```bash
python release/prepare_hf.py --check-only
python release/prepare_hf.py --upload
```

The helper verifies every file's length and SHA256 and checks the expanded model layout before upload. It commits only the six model files, README, LICENSE, NOTICE, the comparison figure, navigation badges, and manifest. It preserves private visibility and uses the current Hub commit as an upload precondition. It does not include training data, optimizer states, or closed evaluation items.

## Validate the loading example

All six inference files have been uploaded and verified. A local Transformers check loaded the checkpoint on CPU in FP32/SDPA and completed one short greedy generation. The selected weights, 24-layer layout, and 497,119,488 parameters match the release manifest. Details are in [the release check record](../evidence/release_validation.json).

The CUDA example in the model card still needs validation in a fresh GPU environment with BF16 support. Record the Hub revision and observed output when running that check; the card keeps this status explicit.

The current model card is recorded in Hugging Face commit `ec9191f9ad266d0024f3f3f5379a6451ac418493` on 28 October 2026, alongside LICENSE, NOTICE, the embedded comparison figure, and navigation badges. All six checkpoint files and the release manifest are uploaded. The card contents, file hashes, and Hub metadata have been verified.

Training data, optimizer states, and closed evaluation items remain outside the model release.
