#!/usr/bin/env python3

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MERGE = HERE / "merge_aot_packages.py"


def _string(value: str) -> bytes:
    encoded = value.encode()
    if len(encoded) < 32:
        return bytes([0xA0 | len(encoded)]) + encoded
    return b"\xD9" + bytes([len(encoded)]) + encoded


def _uint(value: int) -> bytes:
    if value < 128:
        return bytes([value])
    if value < 256:
        return b"\xCC" + bytes([value])
    return b"\xCD" + value.to_bytes(2, "big")


def _array(values) -> bytes:
    values = list(values)
    prefix = bytes([0x90 | len(values)])
    return prefix + b"".join(values)


def _map(values) -> bytes:
    values = list(values)
    prefix = bytes([0x80 | len(values)])
    return prefix + b"".join(_string(key) + value for key, value in values)


def _argument(offset: int, size: int, pointer: bool) -> bytes:
    values = [
        (".offset", _uint(offset)),
        (".size", _uint(size)),
        (".value_kind", _string("global_buffer" if pointer else "by_value")),
    ]
    if pointer:
        values.insert(0, (".address_space", _string("global")))
    return _map(values)


def _hsaco(name: str) -> bytes:
    descriptor = _map([
        ("amdhsa.kernels", _array([_map([
            (".args", _array([
                _argument(0, 8, True), _argument(8, 4, False),
                _argument(16, 8, True), _argument(24, 8, True),
            ])),
            (".group_segment_fixed_size", _uint(0)),
            (".kernarg_segment_size", _uint(32)),
            (".max_flat_workgroup_size", _uint(64)),
            (".name", _string(name)),
            (".private_segment_fixed_size", _uint(0)),
            (".symbol", _string(f"{name}.kd")),
            (".wavefront_size", _uint(32)),
        ])])),
        ("amdhsa.target", _string("amdgcn-amd-amdhsa--gfx1150")),
        ("amdhsa.version", _array([_uint(1), _uint(2)])),
    ])
    note_name = b"AMDGPU\0"
    note = struct.pack("<III", len(note_name), len(descriptor), 0x20)
    note += note_name + b"\0" * ((-len(note_name)) & 3)
    note += descriptor + b"\0" * ((-len(descriptor)) & 3)
    note_offset = 0x100
    image = bytearray(note_offset + len(note))
    image[:16] = b"\x7fELF\x02\x01\x01\x40\x03" + b"\0" * 7
    struct.pack_into("<HHI", image, 16, 3, 224, 1)
    struct.pack_into("<Q", image, 32, 64)
    struct.pack_into("<HH", image, 52, 64, 56)
    struct.pack_into("<H", image, 56, 1)
    struct.pack_into("<IIQQQQQQ", image, 64, 4, 4, note_offset, 0, 0,
                     len(note), len(note), 4)
    image[note_offset:] = note
    return bytes(image)


def _write_package(root: Path, names: str | list[str], package: bool) -> None:
    root.mkdir()
    if isinstance(names, str):
        names = [names]
    kernels = []
    for name in names:
        artifact = root / f"{name}.hsaco"
        artifact.write_bytes(_hsaco(name))
        kernels.append({
            "name": name,
            "symbol": name,
            "file": artifact.name,
            "num_warps": 2,
            "warp_size": 32,
            "block_size": 8,
            "argument_count": 2,
        })
    manifest = {
        "format": 2,
        "arch": "gfx1150",
        "kernels": kernels,
    }
    if package:
        manifest["package"] = {
            "abi": "flagos-amd-aot-v2",
            "backend": "hip",
            "binary_format": "hsaco",
            "provenance": "generated",
        }
    (root / "manifest.json").write_text(json.dumps(manifest))


class MergePackageTest(unittest.TestCase):
    def _merge(self, base: Path, extension: Path, output: Path, tuning="", add_missing=False):
        command = [sys.executable, str(MERGE), "--base-dir", str(base),
                   "--additional-dir", str(extension), "--output-dir", str(output)]
        if tuning:
            command += ["--tuning-profile", tuning]
        if add_missing:
            command += ["--add-missing"]
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_legacy_base_plus_v2_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", True)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["package"]["provenance"], "merged-package")
            self.assertEqual(manifest["package"]["source_scope"], "additional-package-only")
            self.assertEqual(manifest["components"][0]["kind"], "legacy-base")
            self.assertEqual(len(manifest["kernels"]), 2)

    def test_v2_base_plus_legacy_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", False)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["package"]["provenance"], "merged-package")
            self.assertEqual(manifest["components"][0]["kind"], "legacy-extension")

    def test_legacy_plus_legacy_has_no_fabricated_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", False)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertNotIn("package", manifest)

    def test_tuning_profile_rejects_legacy_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", True)
            result = self._merge(
                root / "base", root / "extension", root / "out",
                "gfx1150-wave32-q4ffn-v1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires v2 metadata", result.stderr)

    def test_qwen35_profile_is_known_and_requires_exact_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", True)
            result = self._merge(
                root / "base", root / "extension", root / "out",
                "gfx1150-qwen35-q4km-v2")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete gfx1150-qwen35-q4km-v2 package", result.stderr)
            self.assertNotIn("unsupported tuning profile", result.stderr)

    def test_add_missing_preserves_base_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", ["shared", "base"], True)
            _write_package(root / "extension", ["shared", "extension"], True)
            extension_manifest_path = root / "extension" / "manifest.json"
            extension_manifest = json.loads(extension_manifest_path.read_text())
            extension_manifest["kernels"][0]["block_size"] = 16
            extension_manifest_path.write_text(json.dumps(extension_manifest))
            result = self._merge(
                root / "base", root / "extension", root / "out", add_missing=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(
                [kernel["name"] for kernel in manifest["kernels"]],
                ["shared", "base", "extension"],
            )
            self.assertEqual(manifest["kernels"][0]["block_size"], 8)

    def test_exact_block_contract_requires_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", True)
            manifest_path = root / "extension" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["kernels"][0]["exact_block_size"] = 1
            manifest_path.write_text(json.dumps(manifest))
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact_block_size", result.stderr)


if __name__ == "__main__":
    unittest.main()
