# Nyqo n1 Hugging Face release preparation

Target: [Hxtomi/nyqo](https://huggingface.co/Hxtomi/nyqo). The repository currently remains private.

The prepared model page is [huggingface/README.md](huggingface/README.md). It accompanies the original [LICENSE](huggingface/LICENSE), a [modification notice](huggingface/NOTICE), the [comparison figure](huggingface/assets/nyqo-benchmarks.svg), [navigation badges](huggingface/assets/badges/), and a [file manifest](huggingface/manifest.json).

The six checkpoint files are not checked into Git. Their expected bytes and hashes identify the selected n1 checkpoint. The weights SHA256 is `6fcd99004b78850b5a323224a9a9833c9bcb13faa8eb762092aed8922fcb0078`.

## Stage the checkpoint

Use an existing checkpoint directory containing the six files:

```bash
python release/prepare_hf.py --source-dir /path/to/checkpoint
```

Without `--source-dir`, the helper downloads only those files from the pinned private recovery archive. The archive account needs read access; the destination account needs write access to Hxtomi/nyqo. Standard Hugging Face authentication is used for the destination. A separate archive credential can be supplied through the `NYQO_ARCHIVE_TOKEN` environment variable.

Run this from a workspace on the approved work drive. Downloads and the archive cache are placed beside the release bundle.

## Verify and upload privately

```bash
python release/prepare_hf.py --check-only
python release/prepare_hf.py --upload
```

The helper verifies every file's length and SHA256 and checks the expanded model layout before upload. It commits only the six model files, README, LICENSE, NOTICE, the comparison figure, navigation badges, and manifest. It preserves private visibility and uses the current Hub commit as an upload precondition. It does not include training data, optimizer states, or closed evaluation items.

## Validate the loading example

Run the Python example in the model card with access to Hxtomi/nyqo on a CUDA GPU with BF16 support. Record the Hub revision and observed output. The card currently labels this end-to-end check as pending.

The current model card is available in private Hugging Face commit `21833d3d018342fbda169f9200db38036b992e7c` on 28 October 2026, alongside LICENSE, NOTICE, the embedded comparison figure, and navigation badges. The card contents and Hub metadata have been verified. The six checkpoint files and final release manifest have not yet been uploaded.

The release account could not access the pinned recovery archive. To complete the upload, use a local directory containing the six manifest-listed checkpoint files, or an archive credential with read access. A fresh-GPU download and generation check remains pending.
