#!/usr/bin/env python3
"""Merge per-kernel Triton LLVM IR into a single FlagOS device module.

Triton's dlgpu backend asserts one kernel per compilation unit
(``make_cubin``: ``assert len(names) == 1``), so it emits one ``.cubin`` per
``@triton.jit`` function. Loading thirteen separate modules at backend startup
means thirteen files, thirteen ``cuModuleLoad`` calls and thirteen ``CUmodule``
handles to keep alive.

The Denglin toolchain offers no usable linker for this:

* ``cuLinkAddData`` accepts only ``CU_JIT_INPUT_PTX``; every binary input type
  fails with ``CUDA_ERROR_INVALID_HANDLE``, and Triton emits no PTX here.
* ``dllink`` rejects finished cubins (it wants relocatable objects).
* ``llvm-link``/``dlvm-link`` in the SDK are broken: they report linking each
  input but keep only the first module's functions, even for two-line IR.

What does work is merging the textual IR before ``dlcc`` runs. Each Triton
module is self-contained and only references intrinsics, so concatenating the
function bodies is safe provided that:

* module-level metadata is renumbered per input, since every file restarts at
  ``!0`` and the ``!{ptr @kernel, !"kernel", i32 1}`` / ``maxntidx``
  annotations must survive -- without them ``dlcc`` compiles the code but
  emits no callable entry point;
* ``declare`` and ``attributes`` lines are de-duplicated.

Debug info is dropped: ``!dbg`` references and ``DI*`` nodes carry per-module
scopes that would need full remapping, and they contribute nothing at runtime.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

# Metadata we must carry over. Anything else at module level is debug info.
_KERNEL_ANNOTATION = re.compile(r'^!(\d+)\s*=\s*!\{ptr @(\w+),\s*!"(\w+)",\s*i32 (\d+)\}')
_GLOBAL_DECL = re.compile(r'^@(\w+)\s*=\s*(.*)$')


def _parse_module(path: Path):
    """Split one Triton .llir into the pieces the merged module needs.

    Attribute group ids (``#0``, ``#1``, ...) are local to each module and the
    same id routinely means different things in two kernels, so the caller
    renumbers them; here they are just collected verbatim.
    """
    body: list[str] = []
    annotations: list[tuple[str, str, int]] = []
    declares: set[str] = set()
    attributes: dict[str, str] = {}
    globals_: dict[str, str] = {}

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith('; ModuleID') or line.startswith('source_filename'):
            continue
        if line.startswith('target '):
            continue
        # Debug info is dropped, so the module flags that reference it would
        # leave a dangling "Debug Info Version" of 0 and make LLVM warn.
        if line.startswith('!llvm.module.flags') or line.startswith('!llvm.dbg'):
            continue
        if line.startswith('declare'):
            # libdevice declarations carry !dbg subprogram references; those
            # scopes are not carried over, so drop them here too.
            declares.add(re.sub(r'\s+', ' ', re.sub(r'\s*!dbg !\d+', '', line)))
            continue
        if line.startswith('attributes'):
            match = re.match(r'attributes (#\d+) = (.*)$', re.sub(r'\s+', ' ', line))
            if not match:
                raise RuntimeError(f'{path}: unparsable attribute group: {line}')
            attributes[match.group(1)] = match.group(2)
            continue
        if line.startswith('@'):
            # Module-level globals (in practice only Triton's @global_smem
            # dynamic shared-memory placeholder). One declaration must be
            # shared by all kernels, otherwise the IR redefines the symbol.
            match = _GLOBAL_DECL.match(line)
            if not match:
                raise RuntimeError(f'{path}: unparsable global: {line}')
            globals_[match.group(1)] = re.sub(r'\s+', ' ', line)
            continue
        if line.startswith('!'):
            match = _KERNEL_ANNOTATION.match(line)
            if match:
                annotations.append((match.group(2), match.group(3), int(match.group(4))))
            continue
        # Strip references to metadata we are not carrying over.
        line = re.sub(r',?\s*!dbg !\d+', '', line)
        line = re.sub(r',?\s*!srcloc !\d+', '', line)
        line = re.sub(r',?\s*!tbaa !\d+', '', line)
        body.append(line)

    return body, annotations, declares, attributes, globals_


def _merge_global(name: str, existing: str | None, incoming: str) -> str:
    """Reconcile one global declared by several modules.

    Triton emits @global_smem both with and without ``local_unnamed_addr``.
    That attribute only asserts the address is insignificant, so when the
    modules disagree we keep the conservative form (without it), which is
    what a real linker would do.
    """
    if existing is None or existing == incoming:
        return incoming
    stripped_existing = existing.replace(' local_unnamed_addr', '')
    stripped_incoming = incoming.replace(' local_unnamed_addr', '')
    if stripped_existing == stripped_incoming:
        return stripped_existing
    raise RuntimeError(
        f'incompatible declarations for @{name}:\n  {existing}\n  {incoming}')


def merge(inputs: list[Path], output: Path, triple: str) -> list[str]:
    bodies: list[str] = []
    annotations: list[tuple[str, str, int]] = []
    globals_: dict[str, str] = {}
    kernels: list[str] = []

    # Deduplicate attribute-group bodies across modules and give each a unique
    # id, rewriting every reference in the module that used the old numbering.
    group_ids: dict[str, str] = {}
    all_declares: list[str] = []

    for path in inputs:
        body, notes, decls, attrs, glbs = _parse_module(path)
        if not notes:
            raise RuntimeError(f'{path}: no kernel annotation found')

        remap: dict[str, str] = {}
        for local_id, definition in attrs.items():
            if definition not in group_ids:
                group_ids[definition] = f'#{len(group_ids)}'
            remap[local_id] = group_ids[definition]

        def rewrite(text: str) -> str:
            # Longest ids first so #1 does not partially match #10.
            for local_id in sorted(remap, key=len, reverse=True):
                text = re.sub(rf'(?<![\w#]){re.escape(local_id)}(?![\d])',
                              remap[local_id], text)
            return text

        bodies.append(rewrite('\n'.join(body).strip()))
        all_declares.extend(rewrite(line) for line in decls)
        annotations.extend(notes)
        for name, declaration in glbs.items():
            globals_[name] = _merge_global(name, globals_.get(name), declaration)
        for name, kind, _ in notes:
            if kind != 'kernel':
                continue
            if name in kernels:
                raise RuntimeError(
                    f'{path}: kernel @{name} is already defined by another module')
            kernels.append(name)

    # A declaration must appear once per symbol. After renumbering, the same
    # intrinsic can carry different attribute groups in different modules
    # (e.g. llvm.nvvm.barrier0 as #2 in one and #4 in another), which LLVM
    # rejects as a redefinition. Keep the first declaration of each symbol;
    # attributes on a declaration are advisory for these intrinsics.
    by_symbol: dict[str, str] = {}
    for line in all_declares:
        match = re.search(r'@([\w.]+)\(', line)
        by_symbol.setdefault(match.group(1) if match else line, line)
    declares = set(by_symbol.values())

    lines = [
        '; FlagOS merged device module -- generated by merge_flagos_module.py',
        f'target triple = "{triple}"',
        '',
    ]
    lines.extend(sorted(globals_.values()))
    lines.append('')
    lines.extend(('\n\n'.join(bodies), ''))
    lines.extend(sorted(declares))
    lines.append('')
    for definition, group_id in group_ids.items():
        lines.append(f'attributes {group_id} = {definition}')
    lines.append('')

    # Renumber annotations into one contiguous block and re-attach nvvm.annotations.
    annotation_ids = []
    for index, (name, kind, value) in enumerate(annotations):
        lines.append(f'!{index} = !{{ptr @{name}, !"{kind}", i32 {value}}}')
        annotation_ids.append(f'!{index}')
    lines.insert(3, f'!nvvm.annotations = !{{{", ".join(annotation_ids)}}}')

    output.write_text('\n'.join(lines) + '\n')
    return kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--triple', default='dlgput64-unknown-cuda')
    parser.add_argument('--verify-with', type=Path, default=None,
                        help='path to llvm-as; when given, the merged IR is parsed to validate it')
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()

    kernels = merge(args.inputs, args.output, args.triple)
    print(f'merged {len(args.inputs)} modules -> {args.output} ({len(kernels)} kernels)')
    for name in kernels:
        print(f'  {name}')

    if args.verify_with:
        subprocess.run([str(args.verify_with), str(args.output), '-o', '/dev/null'], check=True)
        print('IR verified')


if __name__ == '__main__':
    main()
