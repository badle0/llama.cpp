import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml-spacemit/ggml-spacemit.cpp"
s = open(p).read()

# 1. device type knob (also replaces an earlier version of this patch)
s, n1 = re.subn(
    r"static enum ggml_backend_dev_type ggml_backend_spacemit_device_get_type\(ggml_backend_dev_t dev\) \{.*?GGML_UNUSED\(dev\);\n\}",
    """static enum ggml_backend_dev_type ggml_backend_spacemit_device_get_type(ggml_backend_dev_t dev) {
    // experiment knob: GGML_SPACEMIT_DEVICE_TYPE=gpu|igpu|accel (default accel)
    static const enum ggml_backend_dev_type type = [] {
        const char * v = getenv("GGML_SPACEMIT_DEVICE_TYPE");
        if (v != nullptr && strcmp(v, "gpu") == 0) {
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        if (v != nullptr && strcmp(v, "igpu") == 0) {
            return GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }();
    return type;
    GGML_UNUSED(dev);
}""", s, count=1, flags=re.S)

# 2. buffer-policy knob: let ops read non-weight tensors from host buffers
helper = """// experiment knob: GGML_SPACEMIT_ACCEPT_HOST=1 lets ops use non-weight tensors in host buffers
static bool ggml_backend_spacemit_accept_host() {
    static const bool v = getenv("GGML_SPACEMIT_ACCEPT_HOST") != nullptr;
    return v;
}

static bool ggml_backend_spacemit_device_supports_op("""
if "ggml_backend_spacemit_accept_host" not in s:
    s = s.replace("static bool ggml_backend_spacemit_device_supports_op(", helper, 1)
old_chk = """        if (!t || !t->buffer) return true;  // unallocated is OK
        return ggml_backend_buffer_is_spacemit(t->buffer);"""
new_chk = """        if (!t || !t->buffer) return true;  // unallocated is OK
        if (ggml_backend_buffer_is_spacemit(t->buffer)) return true;
        return ggml_backend_spacemit_accept_host() && ggml_backend_buffer_is_host(t->buffer) &&
               ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;"""
n2 = s.count(old_chk)
s = s.replace(old_chk, new_chk, 1)
old_bft = "    return buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;\n    GGML_UNUSED(dev);"
new_bft = """    return buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name ||
           (ggml_backend_spacemit_accept_host() && ggml_backend_buft_is_host(buft));
    GGML_UNUSED(dev);"""
n3 = s.count(old_bft)
s = s.replace(old_bft, new_bft, 1)
open(p, "w").write(s)
ok = n1 == 1 and ("accept_host() && ggml_backend_buffer_is_host" in s) and ("accept_host() && ggml_backend_buft_is_host" in s)
print("patched" if ok else f"PATCH FAILED: get_type={n1} check_buf={n2} supports_buft={n3}")
