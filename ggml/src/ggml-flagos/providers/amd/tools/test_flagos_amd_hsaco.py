#!/usr/bin/env python3

import os
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path

from flagos_amd_hsaco import parse_hsaco, validate_hsaco


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
    prefix = bytes([0x90 | len(values)]) if len(values) < 16 else b"\xDC" + len(values).to_bytes(2, "big")
    return prefix + b"".join(values)


def _map(values) -> bytes:
    values = list(values)
    prefix = bytes([0x80 | len(values)]) if len(values) < 16 else b"\xDE" + len(values).to_bytes(2, "big")
    return prefix + b"".join(_string(key) + value for key, value in values)


def _argument(offset: int, size: int, pointer: bool, scalar_address_space=False) -> bytes:
    values = [(".offset", _uint(offset)), (".size", _uint(size)),
              (".value_kind", _string("global_buffer" if pointer else "by_value"))]
    if pointer:
        values.insert(0, (".address_space", _string("global")))
    elif scalar_address_space:
        values.insert(0, (".address_space", _string("global")))
    return _map(values)


def _descriptor(name="kernel", target="gfx1150", scratch_pointer=True,
                scalar_address_space=False, version=(1, 2)) -> bytes:
    arguments = [_argument(0, 8, True), _argument(8, 4, False, scalar_address_space),
                 _argument(16, 8, scratch_pointer), _argument(24, 8, True)]
    kernel = _map([
        (".args", _array(arguments)),
        (".group_segment_fixed_size", _uint(16)),
        (".kernarg_segment_size", _uint(32)),
        (".max_flat_workgroup_size", _uint(64)),
        (".name", _string(name)),
        (".private_segment_fixed_size", _uint(0)),
        (".symbol", _string(f"{name}.kd")),
        (".wavefront_size", _uint(32)),
    ])
    return _map([
        ("amdhsa.kernels", _array([kernel])),
        ("amdhsa.target", _string(f"amdgcn-amd-amdhsa--{target}")),
        ("amdhsa.version", _array(_uint(value) for value in version)),
    ])


def _hsaco(descriptor=None, duplicate_note=False) -> bytes:
    descriptor = _descriptor() if descriptor is None else descriptor
    name = b"AMDGPU\0"
    note = struct.pack("<III", len(name), len(descriptor), 0x20)
    note += name + b"\0" * ((-len(name)) & 3)
    note += descriptor + b"\0" * ((-len(descriptor)) & 3)
    if duplicate_note:
        note += note
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


class HsacoMetadataTest(unittest.TestCase):
    def test_valid_contract(self):
        metadata = parse_hsaco(_hsaco())
        self.assertEqual(metadata.target, "amdgcn-amd-amdhsa--gfx1150")
        self.assertEqual(metadata.name, "kernel")
        self.assertEqual(len(metadata.arguments), 4)
        self.assertEqual(metadata.group_segment_fixed_size, 16)

    def test_wrong_elf_identity(self):
        image = bytearray(_hsaco())
        image[7] = 0
        with self.assertRaisesRegex(RuntimeError, "AMD HSA"):
            parse_hsaco(bytes(image))

    def test_invalid_elf_header_geometry(self):
        image = bytearray(_hsaco())
        struct.pack_into("<H", image, 52, 0)
        with self.assertRaisesRegex(RuntimeError, "program-header"):
            parse_hsaco(bytes(image))
        image = bytearray(_hsaco())
        struct.pack_into("<Q", image, 32, 0)
        with self.assertRaisesRegex(RuntimeError, "program-header"):
            parse_hsaco(bytes(image))

    def test_invalid_utf8(self):
        descriptor = bytearray(_descriptor())
        descriptor[descriptor.index(b"kernel")] = 0xFF
        with self.assertRaisesRegex(RuntimeError, "UTF-8"):
            parse_hsaco(_hsaco(bytes(descriptor)))

    def test_duplicate_metadata_note(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            parse_hsaco(_hsaco(duplicate_note=True))

    def test_invalid_scratch_abi(self):
        with self.assertRaisesRegex(RuntimeError, "scratch"):
            parse_hsaco(_hsaco(_descriptor(scratch_pointer=False)))

    def test_strict_metadata_version_and_argument_kinds(self):
        with self.assertRaisesRegex(RuntimeError, "metadata version"):
            parse_hsaco(_hsaco(_descriptor(version=(1, 3))))
        with self.assertRaisesRegex(RuntimeError, "argument metadata"):
            parse_hsaco(_hsaco(_descriptor(scalar_address_space=True)))
        descriptor = _descriptor().replace(b"by_value", b"bad_kind", 1)
        with self.assertRaisesRegex(RuntimeError, "argument metadata"):
            parse_hsaco(_hsaco(descriptor))

    def test_manifest_cross_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.hsaco"
            path.write_bytes(_hsaco())
            kernel = {
                "name": "kernel", "symbol": "kernel", "argument_count": 2,
                "num_warps": 2, "warp_size": 32,
            }
            validate_hsaco(path, "gfx1150", kernel)
            kernel["argument_count"] = 3
            with self.assertRaisesRegex(RuntimeError, "argument count"):
                validate_hsaco(path, "gfx1150", kernel)
            kernel["argument_count"] = 2
            with self.assertRaisesRegex(RuntimeError, "target mismatch"):
                validate_hsaco(path, "gfx1200", kernel)
            kernel["warp_size"] = True
            with self.assertRaisesRegex(RuntimeError, "launch contract"):
                validate_hsaco(path, "gfx1150", kernel)
            kernel["warp_size"] = 32
            kernel["argument_count"] = "2"
            with self.assertRaisesRegex(RuntimeError, "launch contract"):
                validate_hsaco(path, "gfx1150", kernel)
            kernel["argument_count"] = 2
            kernel["exact_block_size"] = 1
            with self.assertRaisesRegex(RuntimeError, "launch contract"):
                validate_hsaco(path, "gfx1150", kernel)

    def test_cpp_python_acceptance_matches_on_corruptions(self):
        parser = os.environ.get("FLAGOS_AMD_HSACO_CPP_PARSER")
        if not parser:
            self.skipTest("FLAGOS_AMD_HSACO_CPP_PARSER is unset")
        valid = _hsaco()
        invalid_utf8 = bytearray(_descriptor())
        invalid_utf8[invalid_utf8.index(b"kernel")] = 0xFF
        samples = [
            valid,
            _hsaco(duplicate_note=True),
            _hsaco(_descriptor(scratch_pointer=False)),
            _hsaco(_descriptor(version=(1, 3))),
            _hsaco(_descriptor(scalar_address_space=True)),
            _hsaco(_descriptor().replace(b"by_value", b"bad_kind", 1)),
            _hsaco(bytes(invalid_utf8)),
        ]
        for length in (1, 4, 8, 16, 32, 63, 64, 65, 128, 255, len(valid) - 1):
            samples.append(valid[:length])
        for index in range(128):
            mutated = bytearray(valid)
            offset = (index * 131 + 17) % len(mutated)
            mutated[offset] ^= 1 << (index % 8)
            samples.append(bytes(mutated))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.hsaco"
            for sample in samples:
                path.write_bytes(sample)
                try:
                    parse_hsaco(sample)
                    python_accepts = True
                except RuntimeError:
                    python_accepts = False
                result = subprocess.run(
                    [parser, "--parse-hsaco", str(path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=5)
                self.assertIn(result.returncode, (0, 1))
                self.assertEqual(result.returncode == 0, python_accepts)


if __name__ == "__main__":
    unittest.main()
