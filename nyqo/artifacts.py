"""Pinned artifact metadata and integrity checks; no model-library imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "nyqo-n1.json"


def load_manifest(path=DEFAULT_MANIFEST):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "nyqo-checkpoint-v1":
        raise ValueError("Unsupported checkpoint manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
        raise ValueError("The checkpoint revision must be an immutable commit SHA")
    if not isinstance(data.get("repo_id"), str) or "/" not in data["repo_id"]:
        raise ValueError("Missing checkpoint repository")
    folder = data.get("subfolder", "")
    if (not isinstance(folder, str) or "\\" in folder
            or folder.startswith("/")
            or (folder and any(p in ("", ".", "..") for p in folder.split("/")))):
        raise ValueError("Invalid checkpoint subfolder")
    records = data.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Empty checkpoint manifest")
    names = set()
    for record in records:
        name = record.get("name", "")
        if (not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name or ":" in name):
            raise ValueError("Checkpoint entries must be plain filenames")
        if name in names:
            raise ValueError(f"Duplicate checkpoint entry: {name}")
        names.add(name)
        if type(record.get("bytes")) is not int or record["bytes"] <= 0:
            raise ValueError(f"Invalid byte length: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError(f"Invalid SHA256: {name}")
    return data


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, record):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing checkpoint file: {path.name}")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Byte length mismatch: {path.name}")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return path


def verify_checkpoint(directory, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    directory = Path(directory)
    for record in manifest["files"]:
        verify_file(directory / record["name"], record)
    return directory
