---
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-350M
license: other
license_name: lfm1.0
license_link: LICENSE
tags:
- nyqo
- lfm2
- block-expansion
- knowledge-distillation
---

# Nyqo n1

<div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
  <a href="https://github.com/Hxtomi/nyqo" style="display:inline-flex;"><img src="assets/badges/github.svg" alt="GitHub: Code" width="128" height="28" style="display:inline-block; vertical-align:middle; margin:0; height:28px;" /></a>
  <a href="https://huggingface.co/Hxtomi/nyqo" style="display:inline-flex;"><img src="assets/badges/huggingface.svg" alt="Hugging Face: Model card" width="224" height="28" style="display:inline-block; vertical-align:middle; margin:0; height:28px;" /></a>
</div>

**Nyqo n1** is a compact, general-purpose language model with **497M parameters**, built on [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M). It expands the base model from 16 to 24 layers and trains only the eight added layers, while the original weights stay frozen.

Post-training combines supervised fine-tuning on verified answers with distillation from the base model and an earlier Nyqo checkpoint. Corrected and external reference answers receive supervised loss only. The selected checkpoint saw 800 examples from a fixed set of 1,600; the training corpus is private.

[Training method](https://github.com/Hxtomi/nyqo/blob/main/docs/method.md) · [Evaluation record](https://github.com/Hxtomi/nyqo/blob/main/docs/results.md)

| Property | Value |
|:--|:--|
| Parameters | 497,119,488 total; 142,635,520 trained |
| Architecture | LFM2 hybrid convolution–attention, 24 layers |
| Language | English (training and evaluation) |
| Format | Full model in Safetensors |
| Checkpoint identity | [File manifest and hashes](https://github.com/Hxtomi/nyqo/blob/main/artifacts/nyqo-n1.json) |

## Evaluation

![Nyqo n1 benchmark comparison. Nyqo is highlighted; best compact-model scores are underlined; Qwen 4B is shown separately as a larger-model reference.](assets/nyqo-benchmarks.svg)

[Scores as a text table](https://github.com/Hxtomi/nyqo/blob/main/docs/results.md#current-model-comparison)

With a limited compute budget, we evaluated fixed subsets of **ARC-Easy (594 tasks)**, **IFEval (210 tasks)**, and **MBPP+ (100 tasks)**. These are subset scores, not full-benchmark leaderboard results. **FINAL** is our separate internal evaluation of **204 tasks**.

The additional comparison scores are reported by the author as descriptive references; exact checkpoint IDs, revisions, and per-model inference records are not included in the release evidence.

For n1 and the base model, ARC-Easy is scored by normalized answer-choice likelihood, IFEval by strict prompt-level accuracy, and MBPP+ requires passing both the base and additional tests from one greedy response.

n1 improves on the ARC-Easy and MBPP+ subsets, while its IFEval score is lower. The paired 95% confidence intervals for the ARC-Easy and MBPP+ gains lie above zero; the interval for the IFEval difference includes zero. These analyses cover n1 versus the base model; paired intervals for the additional comparison models are not reported.

MBPP+ format compliance rose from 45 to 97 of 100 responses, which may contribute to the gain; that contribution has not been isolated.

On FINAL, n1 scores **56/204**, versus **45/204** for the base model (**+24.44% relative; +5.39 percentage points**). This meets the project's acceptance criteria. The earlier internal bootstrap interval included zero; the updated aggregate summary has no recomputed paired interval. Raw internal item records were not recovered in the checked backups, so the current total cannot be independently regraded from them.

Full protocols, confidence intervals, historical results, and artifact availability are in the [evaluation record](https://github.com/Hxtomi/nyqo/blob/main/docs/results.md).

## Quick start

**Checkpoint upload and a fresh-GPU loading check are pending.** Once the model files are available, use a CUDA GPU with BF16 support. The recorded environment used Python 3.12, PyTorch 2.11.0, and Transformers 5.8.1; pinned dependencies are provided in the [GitHub repository](https://github.com/Hxtomi/nyqo/blob/main/requirements.txt).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Hxtomi/nyqo"

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": "Explain gradient descent in simple terms."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

max_new_tokens = 768
if inputs["input_ids"].shape[1] + max_new_tokens > 2048:
    raise ValueError("Prompt and response must fit within 2,048 tokens.")

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda").eval()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

response = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

This uses the native chat template, BF16/SDPA, and greedy decoding, reserving 768 new tokens within a total budget of 2,048. That is the evaluated generation profile, not the maximum supported context length; IFEval used a separate 4,096-token window.

For file verification, multi-turn JSON input, and extraction of the final answer after a leading `<think>…</think>` block, use the [generation CLI and usage guide](https://github.com/Hxtomi/nyqo/blob/main/docs/usage.md).

## Limitations

- Results cover one post-training seed and one selected checkpoint. They do not establish robustness across training runs or performance on the full public benchmarks.
- Retention is incomplete: the earlier item-level analysis found 19 lost base-model successes. An aggregate gain does not imply improvement on every task or domain.
- The experiments do not isolate the effects of expansion, answer selection, and distillation. There is no same-size fine-tuning control, no ablation without distillation, and no measured inference speed.
- Overlap checks covered training inputs using exact matching and bounded semantic retrieval. They were not exhaustive; training targets were not compared, and overlap with upstream pretraining is unknown.
- Other languages, longer contexts, tool use, long-horizon agent behavior, production reliability, and safety have not been evaluated.

## License and credits

Model files are distributed under the [LFM 1.0 License](LICENSE), inherited from LiquidAI/LFM2.5-350M. See [NOTICE](NOTICE) for modifications and attribution. Nyqo-authored source code and documentation are under [Apache 2.0](https://github.com/Hxtomi/nyqo/blob/main/LICENSE). The training corpus and closed evaluation bank are not released.

Developed by **Artur Markov-Tsoy**, building on Liquid AI's LFM2.5-350M. Evaluation uses ARC, IFEval, and EvalPlus (MBPP+); their materials retain their own licenses. [Citation metadata](https://github.com/Hxtomi/nyqo/blob/main/CITATION.cff).
