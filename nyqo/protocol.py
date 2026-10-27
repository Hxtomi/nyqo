"""Dependency-free validation and the original structural-answer parser."""
from __future__ import annotations

NEW_POSITIONS = [4, 5, 10, 11, 16, 17, 22, 23]
TOTAL_PARAMETERS = 497_119_488


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a nonempty JSON array")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain only role and content")
        if message["role"] not in ("system", "user", "assistant"):
            raise ValueError("Supported roles: system, user, assistant")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be nonempty text")
    if messages[-1]["role"] != "user":
        raise ValueError("The last message must be a user message")
    return messages


def validate_budget(input_tokens, max_new_tokens, max_total_tokens):
    if any(type(x) is not int or x <= 0
           for x in (input_tokens, max_new_tokens, max_total_tokens)):
        raise ValueError("Token counts and limits must be positive integers")
    if input_tokens + max_new_tokens > max_total_tokens:
        raise ValueError(
            f"Context budget exceeded: {input_tokens} input + {max_new_tokens} "
            f"reserved output > {max_total_tokens}; no truncation was applied"
        )


def validate_config(config):
    if config.get("model_type") != "lfm2" or config.get("num_hidden_layers") != 24:
        raise ValueError("Expected the expanded 24-layer LFM2 checkpoint")
    if config.get("nyqo_expansion", {}).get("new_positions") != NEW_POSITIONS:
        raise ValueError("Unexpected Nyqo expansion layout")


def parse_answer(raw):
    """Only a leading structural block is reasoning; literal mentions stay."""
    body = raw.split("<|im_end|>", 1)[0].strip()
    if not body.startswith("<think>"):
        return {"final": body, "reason": "",
                "status": "plain" if body else "missing_final"}
    tail = body[len("<think>"):]
    if "</think>" not in tail:
        return {"final": "", "reason": tail, "status": "unclosed_reason"}
    reason, final = tail.split("</think>", 1)
    if "<think>" in reason or "</think>" in reason:
        return {"final": "", "reason": reason, "status": "nested_reason"}
    final = final.strip()
    if final.startswith(("<think>", "</think>")):
        return {"final": "", "reason": reason, "status": "repeated_reason"}
    return {"final": final, "reason": reason,
            "status": "reason_final" if final else "missing_final"}
