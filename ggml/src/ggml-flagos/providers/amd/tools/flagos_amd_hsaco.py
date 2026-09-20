"""Bounded ELF/AMDGPU MessagePack metadata reader for FlagOS HSACO packages."""

from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path


MAX_HSACO_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CONTAINER_ITEMS = 1 << 20
MAX_STRING_BYTES = 1 << 20
MAX_NESTING = 64


@dataclass(frozen=True)
class HsacoArgument:
    offset: int
    size: int
    value_kind: str
    address_space: str = ""


@dataclass(frozen=True)
class HsacoKernel:
    target: str
    name: str
    symbol: str
    arguments: tuple[HsacoArgument, ...]
    kernarg_segment_size: int
    max_flat_workgroup_size: int
    wavefront_size: int
    group_segment_fixed_size: int
    private_segment_fixed_size: int


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def _take(self, size: int) -> bytes:
        if size < 0 or size > len(self.data) - self.offset:
            raise RuntimeError("truncated MessagePack payload")
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def _byte(self) -> int:
        return self._take(1)[0]

    def _be(self, size: int) -> int:
        return int.from_bytes(self._take(size), "big")

    def map_size(self) -> int:
        tag = self._byte()
        if tag & 0xF0 == 0x80:
            count = tag & 0x0F
        elif tag == 0xDE:
            count = self._be(2)
        elif tag == 0xDF:
            count = self._be(4)
        else:
            raise RuntimeError("expected MessagePack map")
        if count > MAX_CONTAINER_ITEMS // 2:
            raise RuntimeError("MessagePack map exceeds element limit")
        return count

    def array_size(self) -> int:
        tag = self._byte()
        if tag & 0xF0 == 0x90:
            count = tag & 0x0F
        elif tag == 0xDC:
            count = self._be(2)
        elif tag == 0xDD:
            count = self._be(4)
        else:
            raise RuntimeError("expected MessagePack array")
        if count > MAX_CONTAINER_ITEMS:
            raise RuntimeError("MessagePack array exceeds element limit")
        return count

    def string(self) -> str:
        tag = self._byte()
        if tag & 0xE0 == 0xA0:
            size = tag & 0x1F
        elif tag == 0xD9:
            size = self._be(1)
        elif tag == 0xDA:
            size = self._be(2)
        elif tag == 0xDB:
            size = self._be(4)
        else:
            raise RuntimeError("expected MessagePack string")
        if size > MAX_STRING_BYTES:
            raise RuntimeError("MessagePack string exceeds size limit")
        try:
            value = self._take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("invalid UTF-8 in MessagePack string") from error
        if "\0" in value:
            raise RuntimeError("MessagePack string contains NUL")
        return value

    def unsigned(self) -> int:
        tag = self._byte()
        if tag <= 0x7F:
            return tag
        sizes = {0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}
        if tag in sizes:
            return self._be(sizes[tag])
        signed_sizes = {0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8}
        if tag in signed_sizes:
            raw = self._take(signed_sizes[tag])
            value = int.from_bytes(raw, "big", signed=True)
            if value < 0:
                raise RuntimeError("negative metadata integer")
            return value
        raise RuntimeError("expected unsigned MessagePack integer")

    def skip(self, depth: int = 0) -> None:
        if depth >= MAX_NESTING:
            raise RuntimeError("MessagePack nesting limit exceeded")
        if self.offset >= len(self.data):
            raise RuntimeError("truncated MessagePack value")
        tag = self.data[self.offset]
        if tag <= 0x7F or tag >= 0xE0 or tag in (0xC0, 0xC2, 0xC3):
            self.offset += 1
            return
        if tag & 0xE0 == 0xA0 or tag in (0xD9, 0xDA, 0xDB):
            self.string()
            return
        if tag & 0xF0 == 0x90 or tag in (0xDC, 0xDD):
            for _ in range(self.array_size()):
                self.skip(depth + 1)
            return
        if tag & 0xF0 == 0x80 or tag in (0xDE, 0xDF):
            for _ in range(self.map_size()):
                self.skip(depth + 1)
                self.skip(depth + 1)
            return
        self.offset += 1
        fixed = {
            0xCA: 4, 0xCB: 8, 0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8,
            0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8,
            0xD4: 2, 0xD5: 3, 0xD6: 5, 0xD7: 9, 0xD8: 17,
        }
        if tag in fixed:
            self._take(fixed[tag])
            return
        length_size = {0xC4: 1, 0xC5: 2, 0xC6: 4, 0xC7: 1, 0xC8: 2, 0xC9: 4}
        if tag in length_size:
            size = self._be(length_size[tag])
            if tag >= 0xC7:
                size += 1  # extension type byte
            self._take(size)
            return
        raise RuntimeError("reserved or unknown MessagePack tag")


def _unique(mapping: dict, key: str, value):
    if key in mapping:
        raise RuntimeError(f"duplicate AMDGPU metadata field {key}")
    mapping[key] = value


def _argument(reader: _Reader) -> HsacoArgument:
    values = {}
    for _ in range(reader.map_size()):
        key = reader.string()
        if key in (".offset", ".size"):
            _unique(values, key, reader.unsigned())
        elif key in (".value_kind", ".address_space"):
            _unique(values, key, reader.string())
        else:
            reader.skip()
    if not all(key in values for key in (".offset", ".size", ".value_kind")):
        raise RuntimeError("incomplete kernel argument metadata")
    argument = HsacoArgument(
        values[".offset"], values[".size"], values[".value_kind"],
        values.get(".address_space", ""))
    if argument.size <= 0:
        raise RuntimeError("zero-sized kernel argument")
    if argument.value_kind == "global_buffer":
        if argument.address_space != "global":
            raise RuntimeError("invalid global-buffer argument metadata")
    elif argument.value_kind != "by_value" or ".address_space" in values:
        raise RuntimeError("unsupported kernel argument metadata")
    return argument


def _kernel(reader: _Reader) -> dict:
    values = {}
    integer_fields = {
        ".kernarg_segment_size", ".max_flat_workgroup_size", ".wavefront_size",
        ".group_segment_fixed_size", ".private_segment_fixed_size",
    }
    for _ in range(reader.map_size()):
        key = reader.string()
        if key == ".args":
            count = reader.array_size()
            if count > 4096:
                raise RuntimeError("too many kernel arguments")
            _unique(values, key, tuple(_argument(reader) for _ in range(count)))
        elif key in (".name", ".symbol"):
            _unique(values, key, reader.string())
        elif key in integer_fields:
            _unique(values, key, reader.unsigned())
        else:
            reader.skip()
    required = {".args", ".name", ".symbol"} | integer_fields
    if not required.issubset(values):
        raise RuntimeError("incomplete AMDGPU kernel metadata")
    if (not values[".name"] or not values[".symbol"] or
            values[".kernarg_segment_size"] <= 0 or
            values[".max_flat_workgroup_size"] <= 0 or
            values[".wavefront_size"] <= 0):
        raise RuntimeError("invalid AMDGPU kernel metadata")
    previous_end = 0
    for argument in values[".args"]:
        end = argument.offset + argument.size
        if (argument.offset < previous_end or
                end > values[".kernarg_segment_size"]):
            raise RuntimeError("invalid kernarg layout")
        previous_end = end
    if len(values[".args"]) < 2 or any(
            argument.size != 8 or argument.value_kind != "global_buffer" or
            argument.address_space != "global"
            for argument in values[".args"][-2:]):
        raise RuntimeError("invalid Triton scratch argument ABI")
    return values


def _metadata(descriptor: bytes) -> HsacoKernel:
    reader = _Reader(descriptor)
    root = {}
    for _ in range(reader.map_size()):
        key = reader.string()
        if key == "amdhsa.target":
            _unique(root, key, reader.string())
        elif key == "amdhsa.version":
            count = reader.array_size()
            version = tuple(reader.unsigned() for _ in range(count))
            if version != (1, 2):
                raise RuntimeError("unsupported AMDGPU metadata version")
            _unique(root, key, version)
        elif key == "amdhsa.kernels":
            count = reader.array_size()
            if count != 1:
                raise RuntimeError("HSACO must contain exactly one kernel")
            _unique(root, key, _kernel(reader))
        else:
            reader.skip()
    if reader.offset != len(descriptor):
        raise RuntimeError("trailing bytes in AMDGPU metadata")
    if (not root.get("amdhsa.target") or "amdhsa.kernels" not in root or
            "amdhsa.version" not in root):
        raise RuntimeError("incomplete AMDGPU code-object metadata")
    kernel = root["amdhsa.kernels"]
    return HsacoKernel(
        root["amdhsa.target"], kernel[".name"], kernel[".symbol"], kernel[".args"],
        kernel[".kernarg_segment_size"], kernel[".max_flat_workgroup_size"],
        kernel[".wavefront_size"], kernel[".group_segment_fixed_size"],
        kernel[".private_segment_fixed_size"])


def parse_hsaco(data: bytes) -> HsacoKernel:
    if (len(data) < 64 or data[:4] != b"\x7fELF" or data[4:8] != b"\x02\x01\x01\x40"):
        raise RuntimeError("not an ELF64 little-endian AMD HSA code object")
    elf_type, machine, version = struct.unpack_from("<HHI", data, 16)
    program_offset = struct.unpack_from("<Q", data, 32)[0]
    header_size, program_size, program_count = struct.unpack_from("<HHH", data, 52)
    if (elf_type != 3 or machine != 224 or version != 1 or program_size != 56 or
            header_size != 64 or program_offset < 64 or
            program_count <= 0 or program_count == 0xFFFF or program_count > 1024 or
            program_offset > len(data) or program_size * program_count > len(data) - program_offset):
        raise RuntimeError("invalid AMDGPU ELF program-header table")
    result = None
    for index in range(program_count):
        header = program_offset + index * program_size
        segment_type = struct.unpack_from("<I", data, header)[0]
        if segment_type != 4:
            continue
        segment_offset = struct.unpack_from("<Q", data, header + 8)[0]
        segment_size = struct.unpack_from("<Q", data, header + 32)[0]
        if segment_offset > len(data) or segment_size > len(data) - segment_offset:
            raise RuntimeError("invalid PT_NOTE segment")
        offset = segment_offset
        end = segment_offset + segment_size
        while offset < end:
            if end - offset < 12:
                raise RuntimeError("truncated ELF note header")
            name_size, descriptor_size, note_type = struct.unpack_from("<III", data, offset)
            offset += 12
            padded_name = (name_size + 3) & ~3
            padded_descriptor = (descriptor_size + 3) & ~3
            if padded_name > end - offset or padded_descriptor > end - offset - padded_name:
                raise RuntimeError("invalid ELF note size")
            name = data[offset:offset + name_size]
            descriptor = data[offset + padded_name:offset + padded_name + descriptor_size]
            if name == b"AMDGPU\0" and note_type == 0x20:
                if result is not None:
                    raise RuntimeError("duplicate AMDGPU metadata note")
                if not 0 < descriptor_size <= MAX_METADATA_BYTES:
                    raise RuntimeError("invalid AMDGPU metadata note size")
                result = _metadata(descriptor)
            offset += padded_name + padded_descriptor
    if result is None:
        raise RuntimeError("AMDGPU metadata note not found")
    return result


def read_hsaco(path: Path) -> HsacoKernel:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_HSACO_BYTES:
            raise RuntimeError(f"invalid HSACO file size: {path}")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"truncated HSACO file: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        version = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if version(before) != version(after):
            raise RuntimeError(f"HSACO changed while being read: {path}")
    finally:
        os.close(descriptor)
    return parse_hsaco(b"".join(chunks))


def validate_hsaco(path: Path, arch: str, kernel: dict) -> HsacoKernel:
    if not isinstance(path, Path) or not isinstance(arch, str) or not arch or "\0" in arch:
        raise RuntimeError("invalid HSACO validation target")
    if not isinstance(kernel, dict):
        raise RuntimeError("invalid HSACO kernel contract")
    name = kernel.get("name")
    symbol = kernel.get("symbol", name)
    if (not isinstance(name, str) or not name or "\0" in name or
            not isinstance(symbol, str) or not symbol or "\0" in symbol):
        raise RuntimeError("invalid HSACO kernel symbol contract")
    declared_arguments = kernel.get("argument_count", -1)
    exact_block_size = kernel.get("exact_block_size", False)
    num_warps = kernel.get("num_warps")
    warp_size = kernel.get("warp_size")
    if (type(exact_block_size) is not bool or
            type(declared_arguments) is not int or not -1 <= declared_arguments <= 30 or
            type(num_warps) is not int or not 0 < num_warps <= 1024 or
            type(warp_size) is not int or not 0 < warp_size <= 1024 or
            num_warps > 1024 // warp_size):
        raise RuntimeError("invalid HSACO launch contract")
    metadata = read_hsaco(path)
    target_arch = metadata.target.removeprefix("amdgcn-amd-amdhsa--").split(":", 1)[0]
    if not metadata.target.startswith("amdgcn-amd-amdhsa--") or target_arch != arch:
        raise RuntimeError(f"HSACO target mismatch for {path.name}: {metadata.target} != {arch}")
    if metadata.name != symbol or metadata.symbol != f"{symbol}.kd":
        raise RuntimeError(f"HSACO symbol mismatch for {path.name}")
    explicit_arguments = len(metadata.arguments) - 2
    if declared_arguments >= 0 and declared_arguments != explicit_arguments:
        raise RuntimeError(
            f"HSACO argument count mismatch for {path.name}: "
            f"manifest={declared_arguments}, HSACO={explicit_arguments}")
    expected_wave = warp_size
    expected_workgroup = num_warps * expected_wave
    if (metadata.wavefront_size != expected_wave or
            metadata.max_flat_workgroup_size != expected_workgroup):
        raise RuntimeError(
            f"HSACO launch shape mismatch for {path.name}: "
            f"wave={metadata.wavefront_size}, workgroup={metadata.max_flat_workgroup_size}")
    return metadata
