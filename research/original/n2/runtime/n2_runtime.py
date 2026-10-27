"""N2 mixed CE/native-KL runtime. Cloud-only, offline, no evaluation imports.

The N1 identity expansion is reused unchanged. The model core is the native
LFM2 core; the native frozen lm_head is applied only to response predecessor
positions. Non-reentrant checkpointing recomputes token chunks of the exact
full-vocabulary loss during backward, so no N x vocabulary distribution is
retained across the transformer backward pass or written to disk.
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

import copy
import datetime as dt
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/workspace/Nyqo")
HERE = ROOT / "n2/runtime"
B0 = ROOT / "models/LiquidAI--LFM2.5-350M/9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
K500 = ROOT / "models/E497K-update0500"
CANON_COMMIT = "433c5747dcc992eafc2dfd12b5c543ccffaaa223"
SEED = 20261024
DOMAINS = ("source_use", "instructions", "reasoning_math",
           "knowledge_uncertainty", "dialogue_context", "code")
REFERENCES = ("native_K", "native_B0", "corrected_or_external")
NEW = (4, 5, 10, 11, 16, 17, 22, 23)
TOTAL, FROZEN, TRAINABLE = 497119488, 354483968, 142635520
PARENT_WEIGHT_SHA256 = {
    "B0": "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e",
    "K500": "676518a2085c9e2ea9ce06c2f7fb1ae9128570db2cd6de618657e31e7041cd2c",
}
N1_SOURCE_SHA256 = {
    "runtime.py": "2fc16e13d1064dc3a9413aec48a1d07f3e5babe558b37f43eff5559e0299cf94",
    "expanded_runtime.py": "705c67e83f70d47a9199892a1ccede1ca11e703498d451bf2c77bcf2f117fff9",
}
DEFAULT_MIX = {"group_size": 8, "code_per_group": 4,
               "ce_weights": {"code": 1.0, "other": 1.0}, "kl_weight": 1.0,
               "normalization": "group_component_active_tokens_and_group_native_tokens"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = cloud_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = cloud_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cloud_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE.resolve()):
        raise ValueError("Runtime outputs must remain under /workspace/Nyqo/n2/runtime")
    return path


def training_input(path):
    path = Path(path).resolve()
    allowed = ROOT / "n2/data"
    if not path.is_relative_to(allowed):
        raise ValueError("Read training rows only from /workspace/Nyqo/n2/data")
    parts = [part.lower() for part in path.relative_to(allowed).parts]
    if any(any(term in part for term in ("eval", "private", "final204", "dev84"))
           or part in ("dev", "final", "heldout", "held-out", "test") for part in parts):
        raise ValueError("Evaluation/private input is forbidden")
    return path


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def assert_cloud_cuda():
    if not ROOT.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("This execution requires the authorized cloud CUDA host")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if name != "NVIDIA GeForce RTX 3060" or not 11 * 2**30 <= memory <= 13 * 2**30:
        raise RuntimeError(f"Wrong GPU: {name}, {memory} bytes; no automatic fallback")
    return {"name": name, "total_memory_bytes": memory}


def verify_parent_files():
    actual = {"B0": sha(B0 / "model.safetensors"),
              "K500": sha(K500 / "model.safetensors")}
    if actual != PARENT_WEIGHT_SHA256:
        raise AssertionError(("Immutable parent weight hash mismatch", actual))
    return actual


def _n1_expansion():
    for name, expected in N1_SOURCE_SHA256.items():
        if sha(ROOT / "n1" / name) != expected:
            raise AssertionError(f"Vetted N1 source changed: {name}")
    n1_path = str(ROOT / "n1")
    if n1_path not in sys.path:
        sys.path.insert(0, n1_path)
    return importlib.import_module("expanded_runtime")


def is_insert(name):
    parts = name.split(".")
    return len(parts) > 3 and parts[:2] == ["model", "layers"] and int(parts[2]) in NEW


def assert_layout(model, training, expanded=True):
    if expanded:
        assert model.config.num_hidden_layers == 24
        assert model.config.nyqo_expansion["new_positions"] == list(NEW)
        assert sum(p.numel() for p in model.parameters()) == TOTAL
        for index, layer in enumerate(model.model.layers):
            operator = layer.self_attn if layer.is_attention_layer else layer.conv
            assert operator.layer_idx == index
        for name, parameter in model.named_parameters():
            inserted = is_insert(name)
            assert parameter.requires_grad == (training and inserted), name
            assert parameter.dtype == (torch.float32 if inserted else torch.bfloat16), name
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (TRAINABLE if training else 0)
        assert sum(p.numel() for n, p in model.named_parameters() if not is_insert(n)) == FROZEN
    else:
        assert sum(p.numel() for p in model.parameters()) == FROZEN
        assert all(not p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16


def _load_expanded(path, training):
    """Preserve the true saved FP32 insert values, even for frozen K reference."""
    path = Path(path)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    assert model.config.num_hidden_layers == 24
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as state:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(training and is_insert(name))
            if is_insert(name):
                parameter.data = state.get_tensor(name).to(dtype=torch.float32)
    model.config.use_cache = False
    model.train(training)
    assert_layout(model, training)
    return model


def load_student(branch, checkpoint_path=None, device="cuda", activation_checkpointing=True):
    if checkpoint_path is None:
        if branch == "R":
            model = _load_expanded(K500, True)
        elif branch == "C":
            model, _ = _n1_expansion().build_identity(training=True, device="cpu")
        else:
            raise ValueError("Branch must be R or C")
    else:
        model = _load_expanded(cloud_output(checkpoint_path), True)
    assert_layout(model, True)
    if activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    return model, load_tokenizer()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        str(B0), local_files_only=True, trust_remote_code=False)
    assert tokenizer.eos_token_id == 7 and tokenizer.is_fast
    return tokenizer


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_hashes(model, selection="all"):
    return {name: tensor_sha(parameter) for name, parameter in model.named_parameters()
            if selection == "all" or (selection == "frozen" and not parameter.requires_grad)
            or (selection == "trainable" and parameter.requires_grad)}


def assert_separate_storage(*models):
    pointers = {}
    for model_index, model in enumerate(models):
        for name, parameter in model.named_parameters():
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if key in pointers and pointers[key][0] != model_index:
                raise AssertionError(("Model storage shared", pointers[key], (model_index, name)))
            pointers[key] = (model_index, name)


class ReferencePool:
    """Separate immutable CPU snapshots, with at most one reference on CUDA."""
    def __init__(self):
        self.models = {}
        self.forward_calls = {"native_K": 0, "native_B0": 0}
        self.on_cuda = None

    def get(self, kind):
        if kind not in ("native_K", "native_B0"):
            raise ValueError(kind)
        if kind not in self.models:
            if kind == "native_K":
                model = _load_expanded(K500, False)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(B0), local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, attn_implementation="sdpa")
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.config.use_cache = False
                model.eval()
                assert_layout(model, False, expanded=False)
            model.eval()
            self.models[kind] = model
        return self.models[kind]

    @torch.no_grad()
    def features(self, kind, ids, positions):
        if kind == "corrected_or_external":
            return None, None
        if self.on_cuda is not None:
            raise AssertionError("Only one live CUDA reference is permitted")
        model = self.get(kind)
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        self.on_cuda = kind
        try:
            model.to(ids.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                hidden = output.last_hidden_state[0, positions - 1].detach().clone()
            # The head is separate reference storage. Keep only this small frozen
            # projection plus selected hidden states after offloading the model.
            head = model.get_output_embeddings().weight.detach().clone()
            del output
            self.forward_calls[kind] += 1
            return hidden, head
        finally:
            model.to("cpu")
            self.on_cuda = None

    def hashes(self):
        return {kind: parameter_hashes(self.get(kind)) for kind in ("native_K", "native_B0")}

    def assert_frozen(self):
        assert self.on_cuda is None
        for model in self.models.values():
            assert all(not p.requires_grad and p.grad is None and p.device.type == "cpu"
                       for p in model.parameters())


def encode(row, tokenizer, max_length=2048, allow_technical=False):
    required = ("id", "domain", "messages", "final", "reference_kind", "language")
    if any(key not in row for key in required):
        raise ValueError(("Missing training fields", row.get("id")))
    if row["language"] != "en" or row["domain"] not in DOMAINS:
        raise ValueError(("Invalid language/domain", row["id"]))
    if row["reference_kind"] not in REFERENCES:
        raise ValueError(("Invalid reference_kind", row["id"]))
    if row.get("split", "TRAIN").upper() != "TRAIN":
        raise ValueError(("Only TRAIN rows may enter training", row["id"]))
    if row.get("technical_only") and not allow_technical:
        raise ValueError("Synthetic technical rows cannot enter a training run")
    if any(row.get(key) for key in ("reason", "reasoning", "thinking", "reasoning_content")):
        raise ValueError(("Separate reason stream forbidden", row["id"]))
    messages, final = row["messages"], row["final"]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(("Messages must end in user", row["id"]))
    if any(message.get("role") not in ("system", "user", "assistant")
           or not isinstance(message.get("content"), str) for message in messages):
        raise ValueError(("Invalid native messages", row["id"]))
    if any(any(key in message for key in ("reasoning", "thinking", "reasoning_content"))
           for message in messages):
        raise ValueError(("Reason stream in history", row["id"]))
    if not isinstance(final, str) or not final.strip() or final.lstrip().startswith(("<think>", "</think>")):
        raise ValueError(("Missing final or structural reason stream", row["id"]))
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": final}],
                                         tokenize=False, add_generation_prompt=False)
    expected = prefix + final + "<|im_end|>\n"
    if text != expected:
        raise AssertionError(("Native template mismatch", row["id"]))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError(("Context overflow; no truncation allowed", row["id"], len(ids), max_length))
    start, end = len(prefix), len(prefix) + len(final) + len("<|im_end|>")
    mask = [0] * len(ids)
    for index, (left, right) in enumerate(offsets):
        if start <= left < right <= end:
            mask[index] = 1
        elif left < end and right > start:
            raise AssertionError(("Token crosses supervised boundary", row["id"], index,
                                  (left, right), (start, end)))
    positions = [index for index, active in enumerate(mask) if active]
    if not positions or positions[0] == 0 or ids[positions[-1]] != 7:
        raise AssertionError(("Missing final/EOS supervision", row["id"]))
    eos = positions[-1]
    if any(mask[eos + 1:]) or any(mask[:positions[0]]):
        raise AssertionError("Prompt or technical tail supervised")
    if sum(ids[index] == 7 for index in positions) != 1:
        raise ValueError(("Special EOS inside final", row["id"]))
    return {"id": row["id"], "domain": row["domain"], "reference_kind": row["reference_kind"],
            "component": "code" if row["domain"] == "code" else "other",
            "ids": ids, "mask": mask, "positions": positions, "count": len(positions),
            "row_sha256": object_sha(row), "technical_only": bool(row.get("technical_only")),
            "mask_audit": {"prefix_characters": start, "supervised_end_characters": end,
                           "eos_position": eos, "masked_tail_tokens": len(ids) - eos - 1}}


def load_rows(path, expected_sha256):
    path = training_input(path)
    actual = sha(path)
    if not expected_sha256 or actual != expected_sha256:
        raise AssertionError(("Frozen training file SHA256 mismatch", str(path), actual, expected_sha256))
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if isinstance(rows, dict):
            rows = rows["rows"]
    if not isinstance(rows, list) or not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty unique frozen TRAIN rows")
    return rows


def denominators(group):
    result = {"ce": {"code": 0, "other": 0}, "native": 0,
              "by_domain": {domain: {"ce": 0, "native_K": 0, "native_B0": 0,
                                     "corrected_or_external": 0} for domain in DOMAINS},
              "by_component_reference": {component: {kind: 0 for kind in REFERENCES}
                                         for component in ("code", "other")},
              "by_reference": {kind: 0 for kind in REFERENCES}}
    for example in group:
        count, component, kind = example["count"], example["component"], example["reference_kind"]
        assert count == len(example["positions"]) == sum(example["mask"]) and count > 0
        result["ce"][component] += count
        result["by_reference"][kind] += count
        result["by_domain"][example["domain"]]["ce"] += count
        result["by_domain"][example["domain"]][kind] += count
        result["by_component_reference"][component][kind] += count
        if kind != "corrected_or_external":
            result["native"] += count
    return result


def validate_mix(mix):
    if set(mix) != set(DEFAULT_MIX):
        raise ValueError("Explicit mix must specify group_size, code_per_group, ce_weights, kl_weight, normalization")
    if mix["group_size"] != 8:
        raise ValueError("This runtime implements effective batch 8")
    if not 0 <= mix["code_per_group"] <= 8:
        raise ValueError("Invalid code count")
    if set(mix["ce_weights"]) != {"code", "other"} or any(
            not math.isfinite(float(x)) or x < 0 for x in mix["ce_weights"].values()):
        raise ValueError("Invalid CE weights")
    if not math.isfinite(float(mix["kl_weight"])) or mix["kl_weight"] < 0:
        raise ValueError("Invalid KL weight")
    if mix["normalization"] != DEFAULT_MIX["normalization"]:
        raise ValueError("Different normalization requires an explicit runtime revision")
    return mix


def validate_group(group, mix=DEFAULT_MIX):
    validate_mix(mix)
    if len(group) != mix["group_size"] or sum(x["component"] == "code" for x in group) != mix["code_per_group"]:
        raise ValueError("Wrong frozen effective group composition")
    if len({x["id"] for x in group}) != len(group):
        raise ValueError("Duplicate row inside effective group")
    return denominators(group)


def build_order(examples, mix=DEFAULT_MIX, seed=SEED):
    """One shuffled pass; cycle the five other domains before returning to one."""
    validate_mix(mix)
    if len(examples) % 8:
        raise ValueError("Freeze a full multiple of eight; no silent drop/duplication")
    rng = random.Random(seed)
    queues = {domain: [x["id"] for x in examples if x["domain"] == domain] for domain in DOMAINS}
    for queue in queues.values():
        rng.shuffle(queue)
    number = len(examples) // 8
    if len(queues["code"]) != number * mix["code_per_group"]:
        raise ValueError("Code count does not fit explicit group mix")
    other_domains = sorted(DOMAINS[:-1], key=lambda domain: (-len(queues[domain]), DOMAINS.index(domain)))
    if mix == DEFAULT_MIX and max(map(lambda d: len(queues[d]), other_domains)) - min(
            map(lambda d: len(queues[d]), other_domains)) > 1:
        raise ValueError("Small-stage other domains must be uniformly represented")
    groups, cursor = [], 0
    for _ in range(number):
        codes = [queues["code"].pop() for _ in range(mix["code_per_group"])]
        others = []
        for _ in range(8 - mix["code_per_group"]):
            for _attempt in range(len(other_domains)):
                domain = other_domains[cursor % len(other_domains)]
                cursor += 1
                if queues[domain]:
                    others.append(queues[domain].pop())
                    break
            else:
                raise ValueError("Insufficient other examples")
        group = []
        for index in range(max(len(codes), len(others))):
            if index < len(codes):
                group.append(codes[index])
            if index < len(others):
                group.append(others[index])
        groups.append(group)
    assert not any(queues.values())
    assert len({key for group in groups for key in group}) == len(examples)
    return groups


def validate_order(examples, order, mix=DEFAULT_MIX):
    by_id = {example["id"]: example for example in examples}
    flat = [key for group in order for key in group]
    if len(flat) != len(examples) or len(set(flat)) != len(flat) or set(flat) != set(by_id):
        raise ValueError("Frozen order must be one exact pass through frozen rows")
    for group in order:
        validate_group([by_id[key] for key in group], mix)
    return by_id


def corpus_summary(examples, order, mix=DEFAULT_MIX):
    by_id = validate_order(examples, order, mix)
    counts = denominators(examples)
    return {"rows": len(examples), "groups": len(order), "tokens": counts,
            "groups_without_native": sum(denominators([by_id[key] for key in group])["native"] == 0
                                         for group in order),
            "order_sha256": object_sha(order), "encoded_sha256": object_sha(examples),
            "maximum_context_tokens": max(len(x["ids"]) for x in examples),
            "mix": mix, "seed": SEED}


def chunk_numerators(hidden, head, targets, reference_hidden=None, reference_head=None):
    """Native lm_head, T=1, KL(reference || student), full vocabulary sum."""
    logits = F.linear(hidden, head).float()
    ce_sum = F.cross_entropy(logits, targets, reduction="sum")
    if reference_hidden is None:
        kl_sum = ce_sum * 0.0
    else:
        with torch.no_grad():
            reference_logits = F.linear(reference_hidden, reference_head).float()
            reference_logp = F.log_softmax(reference_logits, dim=-1)
        student_logp = F.log_softmax(logits, dim=-1)
        kl_sum = F.kl_div(student_logp, reference_logp, reduction="sum", log_target=True)
    return torch.stack((ce_sum, kl_sum))


def sample_loss(student, example, denoms, references, mix=DEFAULT_MIX, token_chunk_size=32):
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    device = next(student.parameters()).device
    ids = torch.tensor([example["ids"]], device=device, dtype=torch.long)
    positions = torch.tensor(example["positions"], device=device, dtype=torch.long)
    kind = example["reference_kind"]
    reference_hidden, reference_head = references.features(kind, ids, positions)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = student.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        hidden = output.last_hidden_state[0, positions - 1]
        head = student.get_output_embeddings().weight
        loss = hidden.sum() * 0.0
        sums = {"ce": 0.0, "kl": 0.0}
        for begin in range(0, len(positions), token_chunk_size):
            end = min(begin + token_chunk_size, len(positions))
            ref = reference_hidden[begin:end] if reference_hidden is not None else None
            # Checkpoint only the loss/head chunks, not reference forwards. All
            # saved inputs are hidden states/targets or shared frozen weights.
            parts = checkpoint(chunk_numerators, hidden[begin:end], head,
                               ids[0, positions[begin:end]], ref, reference_head,
                               use_reentrant=False)
            loss = loss + mix["ce_weights"][example["component"]] * parts[0] / denoms["ce"][example["component"]]
            if kind != "corrected_or_external":
                assert denoms["native"] > 0
                loss = loss + mix["kl_weight"] * parts[1] / denoms["native"]
            sums["ce"] += float(parts[0].detach())
            sums["kl"] += float(parts[1].detach())
    return loss, sums


def make_optimizer(student):
    return torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                             lr=1e-5, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=0.01, foreach=False)


def make_scheduler(optimizer, horizon, warmup=None):
    horizon = int(horizon)
    warmup = int(math.ceil(0.03 * horizon)) if warmup is None else int(warmup)
    if horizon < 1 or warmup < 0 or warmup > horizon:
        raise ValueError("Invalid explicit scheduler horizon/warmup")
    def factor(completed):
        update = min(completed + 1, horizon)
        if warmup and update <= warmup:
            return update / warmup
        if horizon == warmup:
            return 1.0
        phase = (update - warmup) / (horizon - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * phase))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), {
        "horizon": horizon, "warmup": warmup, "peak_lr": 1e-5,
        "minimum_factor": 0.1, "definition": "next_update_1_based_endpoint_exact"}


def update(student, optimizer, scheduler, group, references, mix=DEFAULT_MIX,
           token_chunk_size=32, audit_gradients=False):
    denoms = validate_group(group, mix)
    optimizer.zero_grad(set_to_none=True)
    ce = {"code": 0.0, "other": 0.0}
    kl_by_reference = {kind: 0.0 for kind in REFERENCES}
    ce_by_domain = {domain: 0.0 for domain in DOMAINS}
    kl_by_component_reference = {component: {kind: 0.0 for kind in REFERENCES} for component in ("code", "other")}
    for example in group:
        loss, parts = sample_loss(student, example, denoms, references, mix, token_chunk_size)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite mixed objective")
        loss.backward()
        ce[example["component"]] += parts["ce"]
        ce_by_domain[example["domain"]] += parts["ce"]
        kl_by_reference[example["reference_kind"]] += parts["kl"]
        kl_by_component_reference[example["component"]][example["reference_kind"]] += parts["kl"]
        del loss
    references.assert_frozen()
    if any(p.grad is not None for p in student.parameters() if not p.requires_grad):
        raise AssertionError("Frozen student parameter received a gradient")
    parameters = [p for p in student.parameters() if p.requires_grad]
    audit = None
    if audit_gradients:
        audit = {name: {"has_gradient": p.grad is not None,
                        "finite": p.grad is not None and bool(torch.isfinite(p.grad).all()),
                        "nonzero": p.grad is not None and bool(torch.count_nonzero(p.grad)),
                        "max_abs": float(p.grad.detach().abs().max()) if p.grad is not None else 0.0}
                 for name, p in student.named_parameters() if p.requires_grad}
        if not all(x["has_gradient"] and x["finite"] for x in audit.values()):
            raise AssertionError("Missing or nonfinite insertion gradient")
        if not any(x["nonzero"] for x in audit.values()):
            raise AssertionError("All insertion gradients are zero")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    components = {part: ce[part] / denoms["ce"][part] if denoms["ce"][part] else 0.0 for part in ce}
    kl_sum = sum(kl_by_reference.values())
    native_kl = kl_sum / denoms["native"] if denoms["native"] else 0.0
    return {"loss": sum(mix["ce_weights"][key] * components[key] for key in ce) + mix["kl_weight"] * native_kl,
            "component_ce": components, "ce_numerators": ce, "ce_by_domain": ce_by_domain,
            "native_kl": native_kl, "kl_numerator": kl_sum,
            "kl_numerators_by_reference": kl_by_reference,
            "kl_numerators_by_component_reference": kl_by_component_reference,
            "token_denominators": denoms, "kl_weight": mix["kl_weight"],
            "grad_norm": float(grad_norm), "lr": lr, "gradient_audit": audit,
            "reference_forward_calls_cumulative": dict(references.forward_calls),
            "normalization": mix["normalization"], "extra_batch_divisor": 1}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def verify_checkpoint(path):
    path = cloud_output(path)
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["status"] != "COMPLETE":
        raise ValueError("Incomplete checkpoint")
    for relative, record in manifest["files"].items():
        file_path = (path / relative).resolve()
        if not file_path.is_relative_to(path):
            raise ValueError("Checkpoint manifest path escape")
        if file_path.stat().st_size != record["bytes"] or sha(file_path) != record["sha256"]:
            raise AssertionError(("Checkpoint file mismatch", relative))
    return manifest


def save_checkpoint(student, tokenizer, optimizer, scheduler, path, state):
    path = cloud_output(path)
    if path.exists():
        raise FileExistsError(f"Never overwrite a checkpoint: {path}")
    path.mkdir(parents=True)
    student.save_pretrained(path, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(path)
    resume = copy.deepcopy(state)
    resume.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state())
    temporary = path / "training_state.pt.tmp"
    torch.save(resume, temporary)
    os.replace(temporary, path / "training_state.pt")
    manifest = {"status": "COMPLETE", "utc": utc(), "canon_commit": CANON_COMMIT,
                "branch": state["branch"], "stage": state["stage"],
                "parent": state["parent"], "completed_updates": state["completed_updates"],
                "next_data_group": state["next_data_group"],
                "exposure_updates": len(state["exposure"]),
                "actual_row_presentations": sum(len(x["ids"]) for x in state["exposure"]),
                "actual_active_tokens": sum(x["active_tokens"] for x in state["exposure"]),
                "scheduler_contract": state["scheduler_contract"],
                "parent_weight_sha256": PARENT_WEIGHT_SHA256,
                "files": {str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                          for p in sorted(path.rglob("*")) if p.is_file()}}
    write_json(path / "manifest.json", manifest)
    return manifest


def restore_training(path, activation_checkpointing=True):
    manifest = verify_checkpoint(path)
    state = torch.load(cloud_output(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["canon_commit"] != CANON_COMMIT:
        raise ValueError("Checkpoint canon differs; explicit migration required")
    student, tokenizer = load_student(state["branch"], path,
                                      activation_checkpointing=activation_checkpointing)
    optimizer = make_optimizer(student)
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler, contract = make_scheduler(optimizer, state["scheduler_contract"]["horizon"],
                                         state["scheduler_contract"]["warmup"])
    assert contract == state["scheduler_contract"]
    scheduler.load_state_dict(state.pop("scheduler"))
    for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        group["lr"] = lr
    restore_rng(state.pop("rng"))
    if state["completed_updates"] != manifest["completed_updates"]:
        raise AssertionError("Resume metadata mismatch")
    return student, tokenizer, optimizer, scheduler, state


def exposure_record(group, metrics, update_index, technical=False):
    return {"update": update_index, "utc": utc(), "ids": [x["id"] for x in group],
            "row_sha256": [x["row_sha256"] for x in group],
            "active_tokens": sum(x["count"] for x in group),
            "tokens": metrics["token_denominators"],
            "technical_only": technical}


def synchronize_timed_update(*args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    metrics = update(*args, **kwargs)
    torch.cuda.synchronize()
    metrics.update(seconds=time.monotonic() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return metrics
