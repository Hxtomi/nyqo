"""Check source bindings, result arithmetic, and local document links offline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nyqo.artifacts import load_manifest, verify_file


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    manifest = load_manifest()
    assert len(manifest["files"]) == 6
    audit = read_json("evidence/closeout.json")
    candidate = audit["candidate"]
    assert candidate["parameters"] == candidate["trainable"] + candidate["frozen"]
    assert candidate["parameters"] == manifest["parameters"] == 497_119_488
    weight = next(x for x in manifest["files"] if x["name"] == "model.safetensors")
    assert candidate["weight_sha256"] == weight["sha256"]
    target = audit["TARGET"]
    for model, total in (("B0", target["B0_actual_runs"][0]),
                         ("C100", target["C100_bounds"])):
        counts = [x[model] for x in target["domain_counts"].values()]
        if model == "B0":
            assert sum(counts) == total
        else:
            assert [sum(x[i] for x in counts) for i in (0, 1)] == total
    rows = {
        "ARC_Easy_acc_norm": (594, 336, 370),
        "IFEval_strict": (210, 159, 153),
        "CODE100_base_plus": (100, 3, 13),
    }
    comparison = read_json("evidence/comparison_summary.json")
    metrics = ("final", "arc_easy", "ifeval", "mbpp_plus")
    models = comparison["models"]
    by_label = {model["label"]: model["passed"] for model in models}
    decisions = comparison["final_scoring_decision"]["changes"]
    assert len({item["model"] for item in decisions}) == len(decisions) == 4
    for item in decisions:
        assert item["selected_passed"] == by_label[item["model"]]["final"]
        assert item["selected_percentage"] == f"{100 * item['selected_passed'] / comparison['task_counts']['final']:.2f}"
    assert next(item["selected_passed"] for item in decisions if item["model"] == "Nyqo n1") == 56
    assert len(by_label) == len(models) == 6
    assert set(comparison["task_counts"]) == set(metrics)
    for model in models:
        assert set(model["passed"]) == set(metrics)
        for metric in metrics:
            assert type(model["passed"][metric]) is int
            assert 0 <= model["passed"][metric] <= comparison["task_counts"][metric]
    metric_keys = {
        "ARC_Easy_acc_norm": "arc_easy",
        "IFEval_strict": "ifeval",
        "CODE100_base_plus": "mbpp_plus",
    }
    for name, (n, baseline, selected) in rows.items():
        result = audit["transfer"][name]
        assert (result["N"], result["B0"], result["C100"]) == (n, baseline, selected)
        key = metric_keys[name]
        assert comparison["task_counts"][key] == n
        assert by_label["LFM2.5-350M (base)"][key] == baseline
        assert by_label["Nyqo n1"][key] == selected

    asset = "assets/nyqo-benchmarks.svg"
    assert (ROOT / asset).read_bytes() == (ROOT / "release/huggingface" / asset).read_bytes()
    figure = ET.parse(ROOT / asset).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    cells = [cell for cell in figure.findall(".//svg:text", ns) if cell.get("data-model")]
    expected_scores = {
        (model["label"], metric): f"{100 * model['passed'][metric] / comparison['task_counts'][metric]:.2f}"
        for model in models for metric in metrics
    }
    actual_scores = {(cell.get("data-model"), cell.get("data-metric")): cell.text for cell in cells}
    assert len(cells) == len(expected_scores)
    assert actual_scores == expected_scores, "Comparison figure scores differ from reported counts"
    compact = [model for model in models if "(reference)" not in model["label"]]
    expected_best = {
        (model["label"], metric)
        for metric in metrics for model in compact
        if model["passed"][metric] == max(item["passed"][metric] for item in compact)
    }
    actual_best = {
        (cell.get("data-model"), cell.get("data-metric"))
        for cell in cells if cell.get("data-best") == "true"
    }
    assert actual_best == expected_best
    assert all(cell.get("font-weight") == ("700" if
               (cell.get("data-model"), cell.get("data-metric")) in expected_best else "400")
               for cell in cells)
    for document in ("README.md", "release/huggingface/README.md"):
        text = (ROOT / document).read_text(encoding="utf-8")
        assert re.search(r"!\[[^\]]+\]\(" + re.escape(asset) + r"\)", text)
        assert "| Model | FINAL (internal)" not in text

    release = read_json("release/huggingface/manifest.json")
    source_by_name = {entry["name"]: entry for entry in manifest["files"]}
    expected_release = set(source_by_name) | {
        "README.md", "LICENSE", "NOTICE", asset,
        "assets/badges/github.svg", "assets/badges/huggingface.svg",
    }
    assert release["repo_id"] == "Hxtomi/nyqo"
    assert len(release["files"]) == len(expected_release)
    assert {entry["name"] for entry in release["files"]} == expected_release
    for entry in release["files"]:
        if entry["name"] in source_by_name:
            assert entry == source_by_name[entry["name"]]
        else:
            verify_file(ROOT / "release/huggingface" / entry["name"], entry)
    provenance = read_json("evidence/provenance.json")
    for entry in provenance["files"]:
        data = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        if "git_blob_sha" in entry:
            blob_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            assert blob_sha == entry["git_blob_sha"], entry["path"]
    missing = []
    for document in ROOT.rglob("*.md"):
        if ".git" in document.parts:
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("#", "https://", "http://", "mailto:")):
                continue
            local = unquote(target.split("#", 1)[0])
            if local and not (document.parent / local).exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)
    print(f"PASS: {len(provenance['files'])} source/receipt bindings, "
          f"checkpoint identity, result counts, and local Markdown links")


if __name__ == "__main__":
    main()
