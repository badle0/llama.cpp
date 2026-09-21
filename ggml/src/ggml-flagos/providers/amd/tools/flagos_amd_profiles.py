"""Shared, dependency-free definitions for known FlagOS AMD AOT profiles."""

import re
from pathlib import Path
from typing import NamedTuple


TUNING_PROFILE_GFX1150_Q4FFN_V1 = "gfx1150-wave32-q4ffn-v1"
TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2 = "gfx1150-qwen35-q4km-v2"
PROFILE_ROOT = Path(__file__).resolve().parent.parent
TUNING_PROFILE_KERNEL_LIST = PROFILE_ROOT / "flagos-amd-tuning-profile.inc"
TUNING_PROFILE_QWEN35_KERNEL_LIST = PROFILE_ROOT / "flagos-amd-tuning-profile-qwen35.inc"
TUNING_PROFILE_KERNEL_LISTS = (
    TUNING_PROFILE_KERNEL_LIST,
    TUNING_PROFILE_QWEN35_KERNEL_LIST,
)


class TunedKernelContract(NamedTuple):
    argument_count: int
    block_size: int
    exact_block_size: bool
    tile_m: int
    tile_n: int
    tile_k: int
    num_warps: int
    warp_size: int


def read_tuning_profile_kernel_abis(
        path: Path, macro: str, has_exact_block_size: bool) -> dict[str, TunedKernelContract]:
    exact_field = r", (true|false)" if has_exact_block_size else ""
    pattern = re.compile(
        rf'{macro}\("([a-z0-9_]+)", ([1-9][0-9]*), '
        rf'([1-9][0-9]*){exact_field}, ([0-9]+), ([0-9]+), ([0-9]+), '
        r'([1-9][0-9]*), ([1-9][0-9]*)\)')
    kernels: dict[str, TunedKernelContract] = {}
    for line_number, source_line in enumerate(path.read_text().splitlines(), start=1):
        line = source_line.strip()
        if not line or line.startswith("//"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise RuntimeError(f"invalid tuning-profile entry at {path}:{line_number}")
        fields = list(match.groups())
        name = fields.pop(0)
        argument_count = int(fields.pop(0))
        block_size = int(fields.pop(0))
        exact_block_size = fields.pop(0) == "true" if has_exact_block_size else False
        contract = TunedKernelContract(
            argument_count, block_size, exact_block_size,
            *(int(value) for value in fields))
        if (name in kernels or contract.argument_count > 30 or
                contract.num_warps > 1024 or
                contract.warp_size > 1024 // contract.num_warps):
            raise RuntimeError(f"invalid tuning-profile ABI at {path}:{line_number}")
        kernels[name] = contract
    if not kernels:
        raise RuntimeError(f"empty or duplicate tuning-profile entries in {path}")
    return kernels


TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNEL_ABIS = read_tuning_profile_kernel_abis(
    TUNING_PROFILE_KERNEL_LIST, "FLAGOS_AMD_TUNED_KERNEL", False)
TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNELS = frozenset(
    TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNEL_ABIS)
TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2_KERNEL_ABIS = read_tuning_profile_kernel_abis(
    TUNING_PROFILE_QWEN35_KERNEL_LIST, "FLAGOS_AMD_TUNED_KERNEL_V2", True)
TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2_KERNELS = frozenset(
    TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2_KERNEL_ABIS)

TUNING_PROFILES = {
    TUNING_PROFILE_GFX1150_Q4FFN_V1: (
        "gfx1150", TUNING_PROFILE_GFX1150_Q4FFN_V1_KERNEL_ABIS),
    TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2: (
        "gfx1150", TUNING_PROFILE_GFX1150_QWEN35_Q4KM_V2_KERNEL_ABIS),
}


def tuning_profile_contracts(name: str) -> tuple[str, dict[str, TunedKernelContract]]:
    try:
        return TUNING_PROFILES[name]
    except KeyError as error:
        raise RuntimeError(f"unsupported AMD tuning profile: {name}") from error
