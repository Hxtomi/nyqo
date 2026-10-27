"""Download the pinned Nyqo n1 inference files and verify every byte."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .artifacts import load_manifest, verify_checkpoint, verify_file


def _hub_fetch(**kwargs):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return hf_hub_download(**kwargs)
    except HfHubHTTPError:
        raise RuntimeError(
            "Checkpoint download failed. Check Hugging Face connectivity and "
            "repository access; private repositories require credentials with read access."
        ) from None


def download_checkpoint(destination, manifest=None, fetch_file=None, cache_dir=None):
    manifest = load_manifest() if manifest is None else manifest
    fetch_file = _hub_fetch if fetch_file is None else fetch_file
    destination = Path(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Checkpoint destination must be a directory")
    # Check all existing files before any download. Never replace a conflicting file.
    for record in manifest["files"]:
        target = destination / record["name"]
        if target.exists() or target.is_symlink():
            verify_file(target, record)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else destination.parent / ".hf-cache"
    for record in manifest["files"]:
        target = destination / record["name"]
        if target.is_file():
            continue
        source = Path(fetch_file(
            repo_id=manifest["repo_id"],
            repo_type=manifest.get("repo_type", "model"),
            revision=manifest["revision"],
            filename="/".join(part for part in
                              (manifest.get("subfolder", ""), record["name"]) if part),
            cache_dir=str(cache_dir),
        ))
        descriptor, temporary_name = tempfile.mkstemp(prefix=".nyqo-", dir=destination)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            verify_file(temporary, record)
            if target.exists() or target.is_symlink():
                verify_file(target, record)
            else:
                os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    verify_checkpoint(destination, manifest)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = download_checkpoint(args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Verified Nyqo n1 checkpoint: {result}")


if __name__ == "__main__":
    main()
