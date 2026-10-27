# Credits and license boundaries

## Nyqo source and documentation

Project author: **Artur Markov-Tsoy**.

Nyqo-authored source code and documentation are released under the **Apache License, Version 2.0**. See the repository [LICENSE](../LICENSE) and [NOTICE](../NOTICE). This grant covers the project code and documentation; model assets and third-party materials retain the separate terms described below.

The archived implementation and new portable interface are distinguished in [source provenance](../evidence/provenance.json). The project builds on established identity-initialized expansion and retention distillation.

## Backbone, tokenizer, and derived checkpoint

Nyqo derives from [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M/tree/9e6c6ccf47cd318696e137d381a7ded8fe4df09f), by Liquid AI. The pinned upstream model card identifies **LFM 1.0 License** (`lfm1.0`), rather than MIT or Apache-2.0.

Read the [upstream license](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/LICENSE) before using or redistributing its weights or derivatives. This repository neither replaces those terms nor claims ownership of upstream weights, tokenizer, or template. The manifest records the archived upstream license SHA256 as `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`.

## Evaluation projects

| Project | Role | Upstream |
|:--|:--|:--|
| ARC / AI2 Reasoning Challenge | ARC-Easy question pool | [allenai/arc](https://github.com/allenai/arc) |
| IFEval | Instruction prompts and official checking | [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) |
| EvalPlus / MBPP+ | Code tasks, base and additional tests | [evalplus/evalplus](https://github.com/evalplus/evalplus) |

Each project's data and code retain their own applicable terms. Dataset contents and checker code are not redistributed under an invented common license here.

## Training and closed evaluation data

The training corpus is private and is not part of this release. Publishing the source code and model weights does not release or license the training archive for redistribution.

Closed N2-FINAL204 prompts, reference material, model responses, and individual grades are excluded. Publishing aggregate counts does not publish that bank.
