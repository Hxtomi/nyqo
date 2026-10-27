# Nyqo n1 — download and generation

Download **Nyqo n1**, verify its weights, and run text generation with the commands below.

## Environment

The recorded environment used a cloud RTX 3060 12 GB, Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 5.8.1, and safetensors 0.7.0. The supplied generation command requires a CUDA GPU with BF16 support. No CPU speed, quantized deployment, or fresh-machine inference result is claimed.

From the repository root:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Use your platform's matching PyTorch CUDA wheel if needed. The requirements file pins direct inference dependencies; it is not a complete transitive training lock.

## Planned Hugging Face release

The release target is [Hxtomi/nyqo](https://huggingface.co/Hxtomi/nyqo), which currently remains private. A Transformers loading example is prepared in the [model card](../release/huggingface/README.md#quick-start); weight upload and end-to-end validation are pending. The archive instructions below describe the current download path and will be updated after the public release checkpoint is verified.

## Obtain the archived checkpoint

The private recovery archive is pinned to a commit, not its moving main branch. You need Hugging Face credentials authorized to read `Exequted/Nyqo`. Standard Hugging Face authentication, including `HF_TOKEN`, is supported by the download library.

~~~bash
python -m nyqo.download --output models/nyqo-n1
~~~

The command downloads only the six files listed in [artifacts/nyqo-n1.json](../artifacts/nyqo-n1.json), verifies byte lengths and SHA256, and does not fetch optimizer states or training data. An existing local file is reused only when its integrity check passes. A conflicting local file causes an error.

If you already have the archived checkpoint, copy its six manifest-listed files into a directory and use that directory directly. This is a full expanded checkpoint, not a LoRA adapter. The weight file is 1,279,535,128 bytes.

Access failure does not mean the archive is missing. The archive was private at this repository's preparation date, 26 October 2026; the public release remains pending.

## Generate an answer

~~~bash
python -m nyqo.generate --checkpoint models/nyqo-n1 \
  --prompt "What is the difference between a parameter and a hyperparameter?"
~~~

The default profile matches the internal generation settings: native chat template, no added system message, greedy decoding, BF16, SDPA, cache enabled, one beam, 768 new tokens, and a total budget of 2,048. Oversized inputs are rejected without truncation. The CLI checks file integrity and model layout before generation and prints the parsed final response.

For a conversation, provide a JSON array of role/content messages ending in a user message:

~~~bash
python -m nyqo.generate --checkpoint models/nyqo-n1 \
  --messages examples/conversation.json
~~~

A different response budget can be passed explicitly:

~~~bash
python -m nyqo.generate --checkpoint models/nyqo-n1 \
  --prompt "Write a short explanation of gradient descent." \
  --max-new-tokens 256 --max-total-tokens 2048 --raw
~~~

Changing limits or prompts defines a different execution profile. `--raw` prints the decoded continuation including structural tokens. No tool calls are executed.

The saved inserted tensors are FP32 to preserve the training state; evaluated generation loads the entire checkpoint in BF16. This convenience command follows the generation profile, not exact optimizer restoration.

## Validation scope

Run the offline checks without installing model dependencies:

~~~bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
~~~

They cover integrity failure handling, pinned download behavior, path validation, context budgets, the answer parser, source hashes, result arithmetic, and local document links. Model libraries are loaded only when a model command actually runs.

The original CPU and CUDA restoration receipts establish that saved weights and training state were recoverable. They did not perform forward/generation/backward/optimizer steps. The new portable CLI has not been run end to end with Nyqo n1 on a fresh GPU machine. Passing the offline suite must not be read as a model-quality rerun.

## Training source

[research/original/](../research/original/) preserves the scientific expansion and training implementation. It retains the original `/workspace/Nyqo` paths, GPU checks, data hashes, and reference dependencies. It is an archival implementation snapshot, not a portable training entry point. Training data, exact order, K500 reference weights, optimizer states, and transfer supplements are tracked separately in [Artifacts](../artifacts/README.md).
