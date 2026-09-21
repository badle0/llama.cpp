#!/usr/bin/env python3
"""Merge compatible FlagOS AMD AOT packages without loading a GPU runtime."""

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from flagos_amd_profiles import (
    TUNING_PROFILES,
    tuning_profile_contracts,
)
from flagos_amd_hsaco import validate_hsaco

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_HSACO_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_BYTES = 4 * 1024 * 1024 * 1024
MAX_KERNELS = 4096
MAX_SIZE_T = (1 << 64) - 1

KNOWN_KERNEL_ABIS = {}
for _, profile_contracts in TUNING_PROFILES.values():
    for kernel_name, kernel_contract in profile_contracts.items():
        previous = KNOWN_KERNEL_ABIS.setdefault(kernel_name, kernel_contract)
        if previous.argument_count != kernel_contract.argument_count:
            raise RuntimeError(f"conflicting tuned kernel ABI contracts for {kernel_name}")


def integer_field(value: dict, name: str, default: int, minimum: int = 0) -> int:
    field = value.get(name, default)
    if isinstance(field, bool) or not isinstance(field, int) or field < minimum:
        raise RuntimeError(f"invalid kernel metadata field {name}")
    return field


def open_regular_file(path: Path, maximum_size: int):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open regular package file: {path}: {error}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum_size:
            raise RuntimeError(f"invalid package file size: {path}")
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def same_file_version(left, right) -> bool:
    return (left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns, left.st_ctime_ns) == (
        right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns, right.st_ctime_ns)


def read_manifest(path: Path) -> bytes:
    descriptor, before = open_regular_file(path, MAX_MANIFEST_BYTES)
    with os.fdopen(descriptor, "rb") as source:
        contents = source.read(MAX_MANIFEST_BYTES + 1)
        after = os.fstat(source.fileno())
    if len(contents) != before.st_size or not same_file_version(before, after):
        raise RuntimeError(f"manifest changed while being read: {path}")
    return contents


def file_integrity(path: Path) -> tuple[int, str]:
    descriptor, before = open_regular_file(path, MAX_HSACO_BYTES)
    digest = hashlib.sha256()
    total = 0
    with os.fdopen(descriptor, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            if total > MAX_HSACO_BYTES:
                raise RuntimeError(f"kernel artifact exceeds size limit: {path}")
            digest.update(chunk)
        after = os.fstat(source.fileno())
    if total != before.st_size or not same_file_version(before, after):
        raise RuntimeError(f"kernel artifact changed while being read: {path}")
    return total, digest.hexdigest()


def copy_validated_artifact(
        source_path: Path, destination_path: Path,
        expected_size: int, expected_sha256: str) -> None:
    descriptor, before = open_regular_file(source_path, MAX_HSACO_BYTES)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "rb") as source, destination_path.open("xb") as destination:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                total += len(chunk)
                if total > MAX_HSACO_BYTES:
                    raise RuntimeError(f"kernel artifact exceeds size limit: {source_path}")
                digest.update(chunk)
                destination.write(chunk)
            after = os.fstat(source.fileno())
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    if (total != before.st_size or not same_file_version(before, after) or
            total != expected_size or digest.hexdigest() != expected_sha256):
        destination_path.unlink(missing_ok=True)
        raise RuntimeError(f"kernel artifact changed after validation: {source_path}")


def reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-standard JSON numeric constant: {value}")


def load_manifest(package_dir: Path, artifacts: dict[str, tuple[int, str]] | None = None) -> dict:
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(
        read_manifest(manifest_path), parse_constant=reject_json_constant)
    if (not isinstance(manifest, dict) or
            type(manifest.get("format")) is not int or manifest["format"] != 2 or
            not isinstance(manifest.get("arch"), str) or not manifest["arch"] or
            "\0" in manifest["arch"] or
            not isinstance(manifest.get("kernels"), list) or
            not 0 < len(manifest["kernels"]) <= MAX_KERNELS):
        raise RuntimeError(f"unsupported manifest in {package_dir}")
    package = manifest.get("package")
    if "package" in manifest and not isinstance(package, dict):
        raise RuntimeError(f"invalid package metadata in {package_dir}")
    if isinstance(package, dict) and (
            package.get("abi") != "flagos-amd-aot-v2" or
            package.get("backend") != "hip" or
            package.get("binary_format") != "hsaco"):
        raise RuntimeError(f"incompatible package metadata in {package_dir}")
    if (isinstance(package, dict) and "tuning_profile" in package and
            (not isinstance(package["tuning_profile"], str) or
             "\0" in package["tuning_profile"])):
        raise RuntimeError(f"invalid tuning profile in {package_dir}")
    tuning_profile = package.get("tuning_profile") if isinstance(package, dict) else None
    if tuning_profile not in (None, "") and tuning_profile not in TUNING_PROFILES:
        raise RuntimeError(f"unsupported tuning profile in {package_dir}")
    if isinstance(package, dict):
        for field in ("provenance", "source_sha256", "generator", "python_version"):
            if (field in package and
                    (not isinstance(package[field], str) or "\0" in package[field])):
                raise RuntimeError(f"invalid package metadata field {field} in {package_dir}")
        if "compiler" in package:
            compiler = package["compiler"]
            if (not isinstance(compiler, dict) or
                    any(field in compiler and
                        (not isinstance(compiler[field], str) or "\0" in compiler[field])
                        for field in ("name", "version"))):
                raise RuntimeError(f"invalid package compiler metadata in {package_dir}")
    tuned = isinstance(package, dict) and bool(package.get("tuning_profile"))
    known_profile = tuning_profile in TUNING_PROFILES
    profile_arch = ""
    profile_contracts = {}
    if known_profile:
        profile_arch, profile_contracts = tuning_profile_contracts(tuning_profile)
    names = set()
    files = set()
    package_bytes = 0
    for kernel in manifest["kernels"]:
        if not isinstance(kernel, dict):
            raise RuntimeError(f"invalid kernel entry in {package_dir}")
        name = kernel.get("name")
        symbol = kernel.get("symbol", name)
        filename = kernel.get("file")
        if (not isinstance(name, str) or not name or
                not isinstance(symbol, str) or not symbol or
                not isinstance(filename, str) or not filename):
            raise RuntimeError(f"invalid kernel entry in {package_dir}")
        if "\0" in name or "\0" in symbol or "\0" in filename:
            raise RuntimeError(f"kernel strings must not contain NUL bytes in {package_dir}")
        if filename in (".", "..") or "/" in filename or "\\" in filename:
            raise RuntimeError(f"kernel artifact must be package-local: {filename}")
        if name in names:
            raise RuntimeError(f"duplicate kernel name in {package_dir}: {name}")
        if filename in files:
            raise RuntimeError(f"duplicate kernel artifact in {package_dir}: {filename}")
        if "arch" in kernel and kernel["arch"] != manifest["arch"]:
            raise RuntimeError(f"kernel architecture mismatch in {package_dir}: {name}")
        shared = integer_field(kernel, "shared", 0)
        num_warps = integer_field(kernel, "num_warps", 0, 1)
        warp_size = integer_field(kernel, "warp_size", 0, 1)
        block_size = integer_field(kernel, "block_size", 0, 1)
        tile_m = integer_field(kernel, "tile_m", 0)
        tile_n = integer_field(kernel, "tile_n", 0)
        tile_k = integer_field(kernel, "tile_k", 0)
        exact_block_size = kernel.get("exact_block_size", False)
        if type(exact_block_size) is not bool:
            raise RuntimeError(
                f"invalid kernel metadata field exact_block_size in {package_dir}: {name}")
        argument_count = kernel.get("argument_count", -1)
        if (type(argument_count) is not int or argument_count > 30 or
                ("argument_count" in kernel and argument_count < 0)):
            raise RuntimeError(f"invalid kernel metadata field argument_count in {package_dir}: {name}")
        global_scratch_size = integer_field(kernel, "global_scratch_size", 0)
        global_scratch_align = integer_field(kernel, "global_scratch_align", 1, 1)
        profile_scratch_size = integer_field(kernel, "profile_scratch_size", 0)
        profile_scratch_align = integer_field(kernel, "profile_scratch_align", 1, 1)
        if (shared > 2**31 - 1 or num_warps > 1024 or
                warp_size > 1024 // num_warps or block_size > 2**31 - 1 or
                tile_m > 2**31 - 1 or tile_n > 2**31 - 1 or tile_k > 2**31 - 1 or
                global_scratch_align > MAX_SIZE_T or profile_scratch_align > MAX_SIZE_T or
                global_scratch_size or profile_scratch_size):
            raise RuntimeError(f"unsupported kernel metadata in {package_dir}: {name}")
        contract = KNOWN_KERNEL_ABIS.get(name)
        if (contract is not None and argument_count >= 0 and
                argument_count != contract.argument_count):
            raise RuntimeError(f"kernel ABI contract mismatch in {package_dir}: {name}")
        if known_profile:
            actual = (argument_count, block_size, exact_block_size,
                      tile_m, tile_n, tile_k, num_warps, warp_size)
            contract = profile_contracts.get(name)
            if contract is None or actual != contract:
                raise RuntimeError(f"tuned kernel launch contract mismatch in {package_dir}: {name}")
        artifact = package_dir / filename
        hsaco = validate_hsaco(artifact, manifest["arch"], kernel)
        if argument_count < 0:
            # A merged legacy package becomes self-describing instead of
            # preserving an unknown call ABI in its newly written manifest.
            argument_count = len(hsaco.arguments) - 2
            kernel["argument_count"] = argument_count
            contract = KNOWN_KERNEL_ABIS.get(name)
            if contract is not None and argument_count != contract.argument_count:
                raise RuntimeError(f"kernel ABI contract mismatch in {package_dir}: {name}")
        actual_size, actual_sha256 = file_integrity(artifact)
        package_bytes += actual_size
        if package_bytes > MAX_PACKAGE_BYTES:
            raise RuntimeError(f"package binaries exceed {MAX_PACKAGE_BYTES} bytes: {package_dir}")
        expected_size = kernel.get("size_bytes")
        if (expected_size is not None and
                (not isinstance(expected_size, int) or isinstance(expected_size, bool) or
                 expected_size != actual_size)):
            raise RuntimeError(f"kernel artifact size mismatch: {artifact}")
        expected_sha256 = kernel.get("sha256")
        if expected_sha256 is not None:
            if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
                    any(character not in "0123456789abcdef" for character in expected_sha256)):
                raise RuntimeError(f"invalid SHA-256 in {package_dir}: {name}")
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"kernel artifact SHA-256 mismatch: {artifact}")
        if tuned and (expected_size is None or expected_sha256 is None):
            raise RuntimeError(f"tuned package kernel lacks integrity metadata: {name}")
        names.add(name)
        files.add(filename)
        if artifacts is not None:
            artifacts[filename] = (actual_size, actual_sha256)
    if known_profile:
        expected_names = frozenset(profile_contracts)
        missing = expected_names - names
        extra = names - expected_names
        if manifest["arch"] != profile_arch or missing or extra:
            detail = "wrong architecture"
            if missing:
                detail = "missing " + ", ".join(sorted(missing))
            elif extra:
                detail = "extra " + ", ".join(sorted(extra))
            raise RuntimeError(
                f"incomplete {tuning_profile} package: {detail}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--additional-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    duplicate_policy = parser.add_mutually_exclusive_group()
    duplicate_policy.add_argument("--replace-existing", action="store_true")
    duplicate_policy.add_argument(
        "--add-missing",
        action="store_true",
        help="add only kernel names absent from the base package",
    )
    parser.add_argument(
        "--tuning-profile",
        default="",
        help="explicitly declare the merged package as a revalidated tuning profile",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="merge only the named kernel from the additional package; may be repeated",
    )
    args = parser.parse_args()

    if args.tuning_profile:
        tuning_profile_contracts(args.tuning_profile)

    base_artifacts: dict[str, tuple[int, str]] = {}
    additional_artifacts: dict[str, tuple[int, str]] = {}
    base = load_manifest(args.base_dir, base_artifacts)
    additional = load_manifest(args.additional_dir, additional_artifacts)
    if base.get("arch") != additional.get("arch"):
        raise RuntimeError(f"architecture mismatch: {base.get('arch')} != {additional.get('arch')}")
    base_package = base.get("package")
    additional_package = additional.get("package")
    legacy_boundary = base_package is None or additional_package is None
    # A number of early FlagOS packages predate manifest-level v2 metadata.
    # They are still safe to extend after every declared HSACO is revalidated;
    # preserve the legacy boundary explicitly instead of fabricating a source
    # hash for artifacts whose generator provenance is unknown.
    if isinstance(base_package, dict) and isinstance(additional_package, dict):
        pass
    elif base_package is None and isinstance(additional_package, dict):
        base_package = copy.deepcopy(additional_package)
        base["package"] = base_package
        base_package["source_scope"] = "additional-package-only"
        base["components"] = [{
            "kind": "legacy-base",
            "arch": base["arch"],
            "kernel_count": len(base["kernels"]),
            "provenance": "unknown",
        }]
    elif isinstance(base_package, dict) and additional_package is None:
        base["components"] = [{
            "kind": "legacy-extension",
            "arch": additional["arch"],
            "kernel_count": len(additional["kernels"]),
            "provenance": "unknown",
        }]
    elif base_package is not None or additional_package is not None:
        raise RuntimeError("invalid package ABI metadata in merge inputs")
    if isinstance(base_package, dict) and isinstance(additional_package, dict):
        for key in ("abi", "backend", "binary_format"):
            if base_package.get(key) != additional_package.get(key):
                raise RuntimeError(
                    f"package {key} mismatch: {base_package.get(key)} != {additional_package.get(key)}")
    if args.kernel:
        requested = set(args.kernel)
        available = {kernel["name"] for kernel in additional["kernels"]}
        missing = requested - available
        if missing:
            raise RuntimeError(
                "selected kernels are absent from the additional package: "
                + ", ".join(sorted(missing)))
        additional["kernels"] = [
            kernel for kernel in additional["kernels"] if kernel["name"] in requested
        ]

    if args.add_missing:
        base_names = {kernel["name"] for kernel in base["kernels"]}
        additional["kernels"] = [
            kernel for kernel in additional["kernels"] if kernel["name"] not in base_names
        ]
        if not additional["kernels"]:
            raise RuntimeError("additional package contains no missing kernels")

    indices = {kernel["name"]: index for index, kernel in enumerate(base["kernels"])}
    file_owners = {kernel["file"]: kernel["name"] for kernel in base["kernels"]}
    for kernel in additional["kernels"]:
        if kernel["name"] in indices and not args.replace_existing:
            raise RuntimeError(f"duplicate kernel: {kernel['name']}")
        owner = file_owners.get(kernel["file"])
        if owner is not None and owner != kernel["name"]:
            raise RuntimeError(
                f"kernel artifact collision: {kernel['file']} belongs to {owner}, not {kernel['name']}")

    if args.output_dir.exists():
        raise RuntimeError(f"output directory already exists: {args.output_dir}")
    if not args.output_dir.parent.is_dir():
        raise RuntimeError(f"output parent directory does not exist: {args.output_dir.parent}")
    if args.tuning_profile and legacy_boundary:
        raise RuntimeError("--tuning-profile requires v2 metadata for every input package")
    if args.tuning_profile and not (
            isinstance(base_package, dict) and isinstance(additional_package, dict)):
        raise RuntimeError("--tuning-profile requires package ABI metadata in both inputs")

    if isinstance(additional_package, dict) and additional_package != base_package:
        components = base.get("components", [])
        if not isinstance(components, list):
            raise RuntimeError("base package components must be an array")
        if additional_package not in components:
            components.append(additional_package)
        base["components"] = components
    if isinstance(base_package, dict):
        base_package["provenance"] = "merged-package"
        base_package.pop("tuning_profile", None)
        if args.tuning_profile:
            base_package["tuning_profile"] = args.tuning_profile

    with tempfile.TemporaryDirectory(
            prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent) as temporary:
        staging_dir = Path(temporary) / "package"
        staging_dir.mkdir()
        # Only declared artifacts cross the package boundary.  Copying the
        # entire source tree would preserve stale HSACOs, symlinks, or other
        # unrelated files that are outside the validated manifest.
        for kernel in base["kernels"]:
            size, digest = base_artifacts[kernel["file"]]
            copy_validated_artifact(
                args.base_dir / kernel["file"], staging_dir / kernel["file"], size, digest)
        for kernel in additional["kernels"]:
            destination = staging_dir / kernel["file"]
            if destination.exists():
                destination.unlink()
            size, digest = additional_artifacts[kernel["file"]]
            copy_validated_artifact(
                args.additional_dir / kernel["file"], destination, size, digest)
            index = indices.get(kernel["name"])
            if index is None:
                indices[kernel["name"]] = len(base["kernels"])
                base["kernels"].append(kernel)
            else:
                previous_file = base["kernels"][index]["file"]
                base["kernels"][index] = kernel
                if previous_file != kernel["file"]:
                    (staging_dir / previous_file).unlink()
        if len(base["kernels"]) > MAX_KERNELS:
            raise RuntimeError(
                f"merged package contains {len(base['kernels'])} kernels; maximum is {MAX_KERNELS}")
        merged_files = set()
        merged_bytes = 0
        for kernel in base["kernels"]:
            filename = kernel["file"]
            if filename in merged_files:
                raise RuntimeError(f"duplicate artifact in merged package: {filename}")
            merged_files.add(filename)
            artifact = staging_dir / filename
            actual_size = artifact.stat().st_size
            merged_bytes += actual_size
            if merged_bytes > MAX_PACKAGE_BYTES:
                raise RuntimeError(
                    f"merged package binaries exceed {MAX_PACKAGE_BYTES} bytes")
        if args.tuning_profile:
            profile_arch, profile_contracts = tuning_profile_contracts(args.tuning_profile)
            expected_names = frozenset(profile_contracts)
            names = {kernel["name"] for kernel in base["kernels"]}
            missing = expected_names - names
            extra = names - expected_names
            if base.get("arch") != profile_arch or missing or extra:
                detail = "wrong architecture"
                if missing:
                    detail = "missing " + ", ".join(sorted(missing))
                elif extra:
                    detail = "extra " + ", ".join(sorted(extra))
                raise RuntimeError(f"incomplete {args.tuning_profile} package: {detail}")
        for kernel in base["kernels"]:
            artifact = staging_dir / kernel["file"]
            kernel["size_bytes"] = artifact.stat().st_size
            kernel["sha256"] = file_integrity(artifact)[1]
        serialized_manifest = json.dumps(base, indent=2, allow_nan=False) + "\n"
        if len(serialized_manifest.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise RuntimeError(
                f"merged manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        (staging_dir / "manifest.json").write_text(serialized_manifest)
        # Re-parse the staged result before publishing it.  This applies the
        # same integrity and package-boundary checks to the final composition,
        # rather than assuming two valid inputs necessarily form a valid sum.
        load_manifest(staging_dir)
        staging_dir.rename(args.output_dir)
    print(f"wrote {len(base['kernels'])} kernels for {base['arch']} to {args.output_dir}")


if __name__ == "__main__":
    main()
