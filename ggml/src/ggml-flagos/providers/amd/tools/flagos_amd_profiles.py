"""Shared, dependency-free definitions for known FlagOS AMD AOT profiles."""

import re
from pathlib import Path
from typing import NamedTuple


TUNING_PROFILE_GFX1150_Q4FFN_V1 = "gfx1150-wave32-q4ffn-v1"
TUNING_PROFILE_KERNEL_LIST = Path(__file__).resolve().parent.parent / "flagos-amd-tuning-profile.inc"


class TunedKernelContract(NamedTuple):
    argument_count: int
    block_size: int
    tile_m: int
    tile_n: int
    tile_k: int
    num_warps: int
    warp_size: int


def read_tuning_profile_kernel_abis(path: Path) -> dict[str, TunedKernelContract]:
    pattern = re.compile(
        r'FLAGOS_AMD_TUNED_KERNEL\("([a-z0-9_]+)", ([1-9][0-9]*), '
        r'([1-9][0-9]*), ([0-9]+), ([0-9]+), ([0-9]+), '
        r'([1-9][0-9]*), ([1-9][0-9]*)\)')
    kernels: dict[str, TunedKernelContract] = {}
    for line_number, source_line in enumerate(path.read_text().splitlines(), start=1):
        line = source_line.strip()
        if not line or line.startswith("//"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise RuntimeError(f"invalid tuning-profile entry at {path}:{line_number}")
        name = match.group(1)
        contract = TunedKernelContract(*(int(value) for value in match.groups()[1:]))
        if (name in kernels or contract.argument_count > 30 or
                contract.num_warps > 1024 or
                contract.warp_size > 1024 // contract.num_warps):
            raise RuntimeError(f"invalid tuning-profile ABI at {path}:{line_number}")
        kernels[name] = contract
    if not kernels:
        raise RuntimeError(f"empty or duplicate tuning-profile entries in {path}")
    return kernels


TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNEL_ABIS = read_tuning_profile_kernel_abis(
    TUNING_PROFILE_KERNEL_LIST)
TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNELS = frozenset(
    TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNEL_ABIS)
