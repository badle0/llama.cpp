#!/usr/bin/env python3
# M0.7 (plan.md): op inventory of a model from llama-eval-callback dumps.
# Lists every op in one graph with weight type, example shapes, weight bytes read and the plan milestone
# that covers it, plus the op sequence of one layer. Pure Python (no numpy): reads the GGUF tensor table itself.
# Usage: op-inventory.py <model.gguf> <name=dump.txt> [<name=dump.txt> ...]
import collections
import os
import re
import struct
import sys

GGML_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1", 10: "q2_K",
              11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K", 15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
              19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32",
              27: "i64", 28: "f64", 29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    """Return (metadata dict, {tensor name: (type name, dims, nbytes)}, check message)."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rd_str():
            return f.read(rd("<Q")).decode("utf-8", errors="replace")

        def rd_val(t):
            if t in SCALAR:
                return rd(SCALAR[t])
            if t == 8:
                return rd_str()
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                items = [rd_val(et) for _ in range(n)]
                return items if n <= 16 else f"<array of {n}>"
            raise ValueError(f"unknown GGUF value type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, n_tensors, n_kv = rd("<I"), rd("<Q"), rd("<Q")
        meta = {}
        for _ in range(n_kv):
            key = rd_str()
            meta[key] = rd_val(rd("<I"))
        infos = []
        for _ in range(n_tensors):
            name = rd_str()
            dims = [rd("<Q") for _ in range(rd("<I"))]
            ttype, offset = rd("<I"), rd("<Q")
            infos.append((name, dims, ttype, offset))
        align = meta.get("general.alignment", 32)
        header_end = f.tell()
        data_start = (header_end + align - 1) // align * align
    size = os.path.getsize(path)
    if not infos:
        return meta, {}, f"GGUF v{version}: no tensors, header ends at {header_end} of {size} bytes"
    ordered = sorted(infos, key=lambda x: x[3])
    tensors = {}
    for i, (name, dims, ttype, offset) in enumerate(ordered):
        end = ordered[i + 1][3] if i + 1 < len(ordered) else size - data_start
        tensors[name] = (GGML_TYPES.get(ttype, f"type{ttype}"), dims, end - offset)
    covered = data_start + sum(t[2] for t in tensors.values())
    check = f"GGUF v{version}: {n_tensors} tensors, data {sum(t[2] for t in tensors.values()) / 2**20:.1f} MiB, " \
            f"header + data = {covered} of {size} bytes ({'ok' if covered == size else 'MISMATCH'})"
    return meta, tensors, check


HEADER = re.compile(r"common_debug_cb_eval:\s+(.+?) = \((\w+)\)\s+(\S+)\((.*)\) = \{([^}]*)\}\s*$")
SRC0 = re.compile(r"^(.*?)\{([^}]*)\}, (.*)$")
SRC1 = re.compile(r"^(.*)\{([^}]*)\}\}$")
LAYER = re.compile(r"-(\d+)(?: \(.*\))?$")
VIEW_OPS = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
BLK = re.compile(r"^blk\.(\d+)\.")
KV_LAYER = re.compile(r"^cache_[kv]_l(\d+)\b")
UNNAMED = re.compile(r"^node_\d+")


def shape(s):
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def read_dump(path):
    nodes = []
    for line in open(path, errors="replace"):
        m = HEADER.search(line)
        if not m:
            continue
        name, otype, op, args, oshape = m.groups()
        a = SRC0.match(args)
        if not a:
            continue
        src0, s0, rest = a.group(1).strip(), shape(a.group(2)), a.group(3)
        b = SRC1.match(rest)
        src1, s1 = (b.group(1).strip(), shape(b.group(2))) if b else ("", [])
        lm = LAYER.search(name)
        nodes.append({"name": name, "otype": otype, "op": op, "src0": src0, "s0": s0, "src1": src1, "s1": s1,
                      "out": shape(oshape), "layer": int(lm.group(1)) if lm else -1})
    # the graph is built and run layer by layer: unnamed nodes (node_N) belong to the current layer,
    # a named tensor without a layer number (inp_embd, norm, result_*) is outside the layers
    cur = -1
    for n in nodes:
        kv = KV_LAYER.search(n["name"])
        wl = BLK.match(n["src0"]) or BLK.match(n["src1"])
        if n["layer"] >= 0:
            cur = n["layer"]
        elif kv:
            cur = int(kv.group(1))
        elif wl:
            cur = int(wl.group(1))
        elif not UNNAMED.match(n["name"]):
            cur = -1
        n["layer"] = cur
    return nodes


def milestone(op, wtype):
    if op in VIEW_OPS:
        return "no kernel (view)"
    if op == "MUL_MAT":
        if wtype == "q4_0":
            return "M2b"
        if wtype:
            return "M2c"
        return "M2e (attention, -fa off)"
    if op == "GET_ROWS" and wtype:
        return "CPU (embedding lookup)"
    if op in ("FLASH_ATTN_EXT", "SOFT_MAX"):
        return "M2e"
    if op in ("RMS_NORM", "NORM", "MUL", "ADD", "SCALE", "ROPE", "SWIGLU", "GEGLU", "SILU", "GELU", "GET_ROWS",
              "SET_ROWS", "CPY", "CONT", "DUP"):
        return "M2a/M2d"
    return "not in plan yet"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: op-inventory.py <model.gguf> <name=dump.txt> [<name=dump.txt> ...]")
    meta, tensors, check = read_gguf(sys.argv[1])
    arch = meta.get("general.architecture", "?")
    print(f"== {os.path.basename(sys.argv[1])}: {arch}, {meta.get(arch + '.block_count', '?')} layers, "
          f"hidden {meta.get(arch + '.embedding_length', '?')}, heads {meta.get(arch + '.attention.head_count', '?')}"
          f"/{meta.get(arch + '.attention.head_count_kv', '?')} (q/kv)")
    print(f"   {check}")
    types = collections.Counter(t[0] for t in tensors.values())
    print("   weight types: " + ", ".join(f"{n} x {t}" for t, n in types.most_common()))

    dumps = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1) if "=" in arg else (os.path.basename(arg), arg)
        nodes = read_dump(path)
        emb = next((n for n in nodes if n["op"] == "GET_ROWS" and n["src0"] in tensors), None)
        n_tok = emb["out"][1] if emb else "?"
        n_view = sum(n["op"] in VIEW_OPS for n in nodes)
        print(f"   dump {label}: {len(nodes)} nodes ({len(nodes) - n_view} compute, {n_view} views), {n_tok} tokens  [{path}]")
        dumps.append((label, nodes))

    rows = collections.OrderedDict()
    for label, nodes in dumps:
        for n in nodes:
            w0 = tensors.get(n["src0"])
            w1 = tensors.get(n["src1"])
            wtype = w0[0] if w0 else ""
            key = (n["op"], wtype or ("w:" + w1[0] if w1 else ""), n["otype"])
            r = rows.setdefault(key, {"count": collections.Counter(), "mib": collections.Counter(), "ex": None})
            r["count"][label] += 1
            if w0:
                r["mib"][label] += w0[2] / 2**20
            if r["ex"] is None:
                r["ex"] = f"{n['s0']} x {n['s1']} -> {n['out']}" if n["s1"] else f"{n['s0']} -> {n['out']}"

    labels = [lb for lb, _ in dumps]
    print()
    print("== ops per graph (count; weight MiB read by that op type; milestone)")
    head = "".join(f"{lb[:12]:>13}" for lb in labels)
    print(f"   {'op':15} {'weight':9} {'out':5}{head}  {'MiB(1st)':>9}  {'milestone':24} example shape (1st dump)")
    for (op, wt, otype), r in sorted(rows.items(), key=lambda kv: (kv[0][0] in VIEW_OPS, -max(kv[1]['mib'].values(), default=0), kv[0])):
        counts = "".join(f"{r['count'][lb]:>13}" for lb in labels)
        mib = r["mib"][labels[0]]
        print(f"   {op:15} {wt:9} {otype:5}{counts}  {mib:9.1f}  {milestone(op, wt if not wt.startswith('w:') else ''):24} {r['ex'][:70]}")

    for label, nodes in dumps:
        mib = sum(tensors[n["src0"]][2] for n in nodes if n["src0"] in tensors and n["op"] == "MUL_MAT") / 2**20
        print(f"   {label}: matrix multiplies read {mib:.1f} MiB of weights per graph")

    label, nodes = dumps[0]
    layer1 = [n for n in nodes if n["layer"] == 1 and n["op"] not in VIEW_OPS]
    if layer1:
        print()
        print(f"== compute ops of layer 1 in order ({label}; views omitted)")
        for n in layer1:
            w = tensors.get(n["src0"])
            print(f"   {n['op']:15} {n['name'][:28]:28} {(w[0] if w else ''):6} {n['s0']} -> {n['out']}")


if __name__ == "__main__":
    main()
