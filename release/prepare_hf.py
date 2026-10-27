"""Stage and verify Nyqo n1 files; optionally upload to its existing private HF repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / "artifacts/nyqo-n1.json").read_text(encoding="utf-8"))
TARGET = "Hxtomi/nyqo"
PAYLOAD_NAMES = {x["name"] for x in SOURCE["files"]}
DOCUMENT_NAMES = {"README.md", "LICENSE", "NOTICE"}
ASSET_NAMES = {
    "assets/nyqo-benchmarks.svg",
    "assets/badges/github.svg",
    "assets/badges/huggingface.svg",
}


def verify_file(path, entry):
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Wrong byte length: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent / "huggingface")
    parser.add_argument("--source-dir", type=Path, help="Existing six-file checkpoint directory")
    parser.add_argument("--check-only", action="store_true", help="Verify without downloading or uploading")
    parser.add_argument("--upload", action="store_true", help="Commit verified files to the private HF repo")
    args = parser.parse_args()
    if args.check_only and args.upload:
        parser.error("--check-only and --upload cannot be combined")
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    allowed_names = PAYLOAD_NAMES | DOCUMENT_NAMES | ASSET_NAMES
    if (manifest["repo_id"] != TARGET or len(entries) != len(allowed_names)
            or {x["name"] for x in entries} != allowed_names):
        raise ValueError("Unexpected destination or release file list")
    for entry in entries:
        if entry["name"] not in ASSET_NAMES and (
                Path(entry["name"]).name != entry["name"] or "/" in entry["name"] or "\\" in entry["name"]):
            raise ValueError("Release entries must be plain filenames")
    source_by_name = {x["name"]: x for x in SOURCE["files"]}
    for entry in entries:
        if entry["name"] in source_by_name and entry != source_by_name[entry["name"]]:
            raise ValueError("Payload hashes differ from the pinned source manifest")
        if entry["name"] in DOCUMENT_NAMES | ASSET_NAMES:
            verify_file(bundle / entry["name"], entry)

    for entry in SOURCE["files"]:
        destination = bundle / entry["name"]
        if destination.exists() or args.check_only:
            verify_file(destination, entry)
            continue
        if args.source_dir:
            source = args.source_dir / entry["name"]
        else:
            cache = bundle.parent / ".archive-cache"
            os.environ.setdefault("HF_HOME", str(cache / "hf"))
            os.environ.setdefault("HF_XET_CACHE", str(cache / "xet"))
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(
                SOURCE["repo_id"], entry["name"], subfolder=SOURCE["subfolder"],
                revision=SOURCE["revision"], cache_dir=cache / "hub",
                token=(os.environ.get("NYQO_SOURCE_TOKEN")
                       or os.environ.get("NYQO_ARCHIVE_TOKEN") or None),
            ))
        verify_file(source, entry)
        shutil.copyfile(source, destination)
        verify_file(destination, entry)

    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "lfm2" or config.get("num_hidden_layers") != 24:
        raise ValueError("Expected the selected 24-layer LFM2 checkpoint")
    if config.get("nyqo_expansion", {}).get("new_positions") != SOURCE["new_positions"]:
        raise ValueError("Unexpected inserted-layer layout")
    print(f"Verified all {len(entries)} release files and the checkpoint layout.")
    if args.upload:
        from huggingface_hub import HfApi, ModelCard
        ModelCard.load(bundle / "README.md")
        api = HfApi()
        before = api.model_info(TARGET)
        if not before.private:
            raise ValueError("This preparation command requires the target to remain private")
        allowed = [x["name"] for x in entries] + ["manifest.json"]
        commit = api.upload_folder(
            repo_id=TARGET, repo_type="model", folder_path=bundle,
            allow_patterns=allowed, parent_commit=before.sha,
            commit_message="Prepare Nyqo n1 model files and card",
        )
        after = api.model_info(TARGET, revision=commit.oid)
        if not after.private:
            raise RuntimeError("Unexpected repository visibility after upload")
        if not set(allowed).issubset({x.rfilename for x in after.siblings}):
            raise RuntimeError("Uploaded commit is missing release files")
        print(f"Uploaded private preparation commit: {commit.oid}")


if __name__ == "__main__":
    main()
