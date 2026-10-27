import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nyqo.artifacts import load_manifest, verify_checkpoint, verify_file
from nyqo.download import download_checkpoint


def record(name, payload):
    return {"name": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def manifest(records):
    return {"schema": "nyqo-checkpoint-v1", "repo_id": "owner/private",
            "repo_type": "model", "revision": "a" * 40,
            "subfolder": "snapshot/payload", "files": records}


class IntegrityTests(unittest.TestCase):
    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_file(path, record(path.name, b"right"))

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "Byte length"):
                verify_file(path, record(path.name, b"longer"))

    def test_missing_checkpoint_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data = manifest([record("config.json", b"{}")])
            with self.assertRaisesRegex(ValueError, "Missing"):
                verify_checkpoint(folder, data)

    def test_invalid_manifest_paths_and_unpinned_revision(self):
        valid = manifest([record("config.json", b"{}")])
        mutations = [
            ("name", "../config.json"), ("name", r"..\config.json"),
            ("name", "C:config.json"), ("revision", "main"),
            ("subfolder", "../payload"), ("subfolder", r"C:\payload"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            for field, value in mutations:
                data = copy.deepcopy(valid)
                if field == "name":
                    data["files"][0][field] = value
                else:
                    data[field] = value
                path.write_text(json.dumps(data))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    load_manifest(path)

    def test_duplicate_manifest_entries_are_rejected(self):
        item = record("config.json", b"{}")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest([item, item])))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_manifest(path)


class DownloadTests(unittest.TestCase):
    def test_download_pins_commit_and_verifies_every_member(self):
        payloads = {"config.json": b"{}", "model.safetensors": b"weights"}
        data = manifest([record(name, blob) for name, blob in payloads.items()])
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "fake-remote"
            cache.mkdir()
            for name, blob in payloads.items():
                (cache / name).write_bytes(blob)

            def fetch(**kwargs):
                calls.append(kwargs)
                return cache / kwargs["filename"].split("/")[-1]

            destination = root / "model"
            download_checkpoint(destination, data, fetch_file=fetch)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(c["revision"] == "a" * 40 for c in calls))
            self.assertTrue(all(c["repo_id"] == "owner/private" for c in calls))
            self.assertEqual({c["filename"] for c in calls},
                             {"snapshot/payload/" + name for name in payloads})
            for name, blob in payloads.items():
                self.assertEqual((destination / name).read_bytes(), blob)
            calls.clear()
            download_checkpoint(destination, data, fetch_file=fetch)
            self.assertEqual(calls, [])  # Idempotent only after integrity validation.

    def test_existing_conflict_stops_before_network_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            target = destination / "config.json"
            target.write_bytes(b"bad")
            data = manifest([record("model.safetensors", b"weights"),
                             record("config.json", b"{}")])

            def forbidden_fetch(**kwargs):
                self.fail("Network must not be used after a local conflict")

            with self.assertRaises(ValueError):
                download_checkpoint(destination, data, fetch_file=forbidden_fetch)
            self.assertEqual(target.read_bytes(), b"bad")
            self.assertFalse((destination / "model.safetensors").exists())

    def test_corrupt_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "remote"
            source.write_bytes(b"bad")
            destination = root / "model"
            data = manifest([record("config.json", b"{}")])
            with self.assertRaises(ValueError):
                download_checkpoint(destination, data, fetch_file=lambda **kw: source)
            self.assertEqual(list(destination.iterdir()), [])

    def test_existing_file_is_not_replaced_with_a_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model"
            target.write_bytes(b"user file")
            with self.assertRaisesRegex(ValueError, "directory"):
                download_checkpoint(target, manifest([record("config.json", b"{}")]))
            self.assertEqual(target.read_bytes(), b"user file")


if __name__ == "__main__":
    unittest.main()
