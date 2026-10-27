"""Nyqo n1 generation using its recorded native BF16/SDPA profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .artifacts import verify_checkpoint
from .protocol import (
    TOTAL_PARAMETERS, parse_answer, validate_budget, validate_config,
    validate_messages,
)


def generate(checkpoint, messages, max_new_tokens=768, max_total_tokens=2048):
    validate_messages(messages)
    validate_budget(1, max_new_tokens, max_total_tokens)
    checkpoint = verify_checkpoint(checkpoint)
    validate_config(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Generation requires a CUDA GPU with BF16 support")
    torch.set_num_threads(4)
    torch.manual_seed(221026)
    torch.cuda.manual_seed_all(221026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 7 or tokenizer.pad_token_id != 0:
        raise ValueError("Unexpected tokenizer EOS/PAD IDs")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_tokens = inputs["input_ids"].shape[1]
    validate_budget(input_tokens, max_new_tokens, max_total_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    if sum(p.numel() for p in model.parameters()) != TOTAL_PARAMETERS:
        raise ValueError("Unexpected parameter count")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("Expected tied input/output embeddings")
    model.config.use_cache = True
    model.eval().to("cuda")
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            use_cache=True, eos_token_id=7, pad_token_id=0)
    token_ids = output[0, input_tokens:].tolist()
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {
        "raw_text": raw, **parse_answer(raw),
        "input_tokens": input_tokens, "output_tokens": len(token_ids),
        "eos_reached": bool(token_ids and token_ids[-1] == 7),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--messages", type=Path, help="JSON role/content array")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--raw", action="store_true", help="Print unparsed continuation")
    args = parser.parse_args()
    try:
        messages = (json.loads(args.messages.read_text(encoding="utf-8"))
                    if args.messages else [{"role": "user", "content": args.prompt}])
        result = generate(args.checkpoint, messages, args.max_new_tokens, args.max_total_tokens)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    if not args.raw and not result["final"]:
        print(f"No parsed final answer (status: {result['status']}); use --raw to inspect.",
              file=sys.stderr)
    print(result["raw_text"] if args.raw else result["final"])


if __name__ == "__main__":
    main()
