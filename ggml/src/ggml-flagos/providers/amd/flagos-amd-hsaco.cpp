#include "flagos-amd-hsaco.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

static constexpr size_t ELF64_HEADER_SIZE = 64;
static constexpr size_t ELF64_PROGRAM_HEADER_SIZE = 56;
static constexpr uint16_t ET_DYN = 3;
static constexpr uint16_t EM_AMDGPU = 224;
static constexpr uint32_t PT_NOTE = 4;
static constexpr uint32_t NT_AMDGPU_METADATA = 0x20;
static constexpr uint8_t ELFCLASS64 = 2;
static constexpr uint8_t ELFDATA2LSB = 1;
static constexpr uint8_t ELFOSABI_AMDGPU_HSA = 64;
static constexpr size_t MAX_METADATA_NOTE_BYTES = 4 * 1024 * 1024;
static constexpr uint64_t MAX_CONTAINER_ITEMS = 1 << 20;
static constexpr uint64_t MAX_STRING_BYTES = 1 << 20;
static constexpr unsigned int MAX_NESTING = 64;

template <typename T>
static bool read_le(const std::vector<uint8_t> & bytes, size_t offset, T & value) {
    static_assert(std::numeric_limits<T>::is_integer, "integer required");
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        return false;
    }
    using unsigned_type = typename std::make_unsigned<T>::type;
    unsigned_type result = 0;
    for (size_t index = 0; index < sizeof(T); ++index) {
        result |= static_cast<unsigned_type>(bytes[offset + index]) << (index * 8);
    }
    value = static_cast<T>(result);
    return true;
}

static bool bounded_region(size_t offset, uint64_t length, size_t limit) {
    return offset <= limit && length <= static_cast<uint64_t>(limit - offset);
}

static bool align4(size_t value, size_t & aligned) {
    if (value > std::numeric_limits<size_t>::max() - 3) {
        return false;
    }
    aligned = (value + 3) & ~size_t(3);
    return true;
}

static bool valid_utf8(const uint8_t * data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const uint8_t first = data[offset++];
        if (first <= 0x7f) {
            continue;
        }
        size_t continuation = 0;
        uint8_t second_min = 0x80;
        uint8_t second_max = 0xbf;
        if (first >= 0xc2 && first <= 0xdf) {
            continuation = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            continuation = 2;
            if (first == 0xe0) second_min = 0xa0;
            if (first == 0xed) second_max = 0x9f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            continuation = 3;
            if (first == 0xf0) second_min = 0x90;
            if (first == 0xf4) second_max = 0x8f;
        } else {
            return false;
        }
        if (continuation > size - offset || data[offset] < second_min || data[offset] > second_max) {
            return false;
        }
        ++offset;
        for (size_t index = 1; index < continuation; ++index, ++offset) {
            if (data[offset] < 0x80 || data[offset] > 0xbf) {
                return false;
            }
        }
    }
    return true;
}

class msgpack_reader {
public:
    msgpack_reader(const uint8_t * begin, size_t size) : current_(begin), end_(begin + size) {}

    bool complete() const { return current_ == end_; }
    const std::string & error() const { return error_; }

    bool read_map_size(uint64_t & size) {
        uint8_t tag = 0;
        if (!byte(tag)) {
            return fail("truncated map");
        }
        if ((tag & 0xf0) == 0x80) {
            size = tag & 0x0f;
        } else if (tag == 0xde) {
            uint16_t value = 0;
            if (!big_endian(value)) return fail("truncated map16");
            size = value;
        } else if (tag == 0xdf) {
            uint32_t value = 0;
            if (!big_endian(value)) return fail("truncated map32");
            size = value;
        } else {
            return fail("expected map");
        }
        return check_container(size, true);
    }

    bool read_array_size(uint64_t & size) {
        uint8_t tag = 0;
        if (!byte(tag)) {
            return fail("truncated array");
        }
        if ((tag & 0xf0) == 0x90) {
            size = tag & 0x0f;
        } else if (tag == 0xdc) {
            uint16_t value = 0;
            if (!big_endian(value)) return fail("truncated array16");
            size = value;
        } else if (tag == 0xdd) {
            uint32_t value = 0;
            if (!big_endian(value)) return fail("truncated array32");
            size = value;
        } else {
            return fail("expected array");
        }
        return check_container(size, false);
    }

    bool read_string(std::string & value) {
        uint8_t tag = 0;
        if (!byte(tag)) {
            return fail("truncated string");
        }
        uint64_t length = 0;
        if ((tag & 0xe0) == 0xa0) {
            length = tag & 0x1f;
        } else if (tag == 0xd9) {
            uint8_t parsed = 0;
            if (!byte(parsed)) return fail("truncated str8");
            length = parsed;
        } else if (tag == 0xda) {
            uint16_t parsed = 0;
            if (!big_endian(parsed)) return fail("truncated str16");
            length = parsed;
        } else if (tag == 0xdb) {
            uint32_t parsed = 0;
            if (!big_endian(parsed)) return fail("truncated str32");
            length = parsed;
        } else {
            return fail("expected string");
        }
        if (length > MAX_STRING_BYTES || length > static_cast<uint64_t>(end_ - current_)) {
            return fail("invalid string length");
        }
        if (!valid_utf8(current_, static_cast<size_t>(length))) {
            return fail("invalid UTF-8 in string");
        }
        value.assign(reinterpret_cast<const char *>(current_), static_cast<size_t>(length));
        current_ += length;
        if (value.find('\0') != std::string::npos) {
            return fail("string contains NUL");
        }
        return true;
    }

    bool read_unsigned(uint64_t & value) {
        uint8_t tag = 0;
        if (!byte(tag)) {
            return fail("truncated integer");
        }
        if (tag <= 0x7f) {
            value = tag;
            return true;
        }
        switch (tag) {
            case 0xcc: { uint8_t parsed = 0; if (!byte(parsed)) break; value = parsed; return true; }
            case 0xcd: { uint16_t parsed = 0; if (!big_endian(parsed)) break; value = parsed; return true; }
            case 0xce: { uint32_t parsed = 0; if (!big_endian(parsed)) break; value = parsed; return true; }
            case 0xcf: { uint64_t parsed = 0; if (!big_endian(parsed)) break; value = parsed; return true; }
            case 0xd0: {
                uint8_t parsed = 0;
                if (!byte(parsed)) break;
                if (parsed > 0x7f) return fail("negative integer");
                value = parsed;
                return true;
            }
            case 0xd1: {
                uint16_t parsed = 0;
                if (!big_endian(parsed)) break;
                if (parsed > 0x7fff) return fail("negative integer");
                value = parsed;
                return true;
            }
            case 0xd2: {
                uint32_t parsed = 0;
                if (!big_endian(parsed)) break;
                if (parsed > 0x7fffffffU) return fail("negative integer");
                value = parsed;
                return true;
            }
            case 0xd3: {
                uint64_t parsed = 0;
                if (!big_endian(parsed)) break;
                if (parsed > 0x7fffffffffffffffULL) return fail("negative integer");
                value = parsed;
                return true;
            }
            default:
                return fail("expected unsigned integer");
        }
        return fail("truncated integer payload");
    }

    bool skip(unsigned int depth = 0) {
        if (depth >= MAX_NESTING) {
            return fail("MessagePack nesting limit exceeded");
        }
        uint8_t tag = 0;
        if (!peek(tag)) {
            return fail("truncated value");
        }
        if (tag <= 0x7f || tag >= 0xe0 || tag == 0xc0 || tag == 0xc2 || tag == 0xc3) {
            ++current_;
            return true;
        }
        if ((tag & 0xe0) == 0xa0 || tag == 0xd9 || tag == 0xda || tag == 0xdb) {
            std::string ignored;
            return read_string(ignored);
        }
        if ((tag & 0xf0) == 0x90 || tag == 0xdc || tag == 0xdd) {
            uint64_t count = 0;
            if (!read_array_size(count)) return false;
            for (uint64_t index = 0; index < count; ++index) {
                if (!skip(depth + 1)) return false;
            }
            return true;
        }
        if ((tag & 0xf0) == 0x80 || tag == 0xde || tag == 0xdf) {
            uint64_t count = 0;
            if (!read_map_size(count)) return false;
            for (uint64_t index = 0; index < count; ++index) {
                if (!skip(depth + 1) || !skip(depth + 1)) return false;
            }
            return true;
        }
        ++current_;
        uint64_t payload = 0;
        switch (tag) {
            case 0xc4: { uint8_t value = 0; if (!byte(value)) return fail("truncated bin8"); payload = value; break; }
            case 0xc5: { uint16_t value = 0; if (!big_endian(value)) return fail("truncated bin16"); payload = value; break; }
            case 0xc6: { uint32_t value = 0; if (!big_endian(value)) return fail("truncated bin32"); payload = value; break; }
            case 0xc7: { uint8_t value = 0; if (!byte(value)) return fail("truncated ext8"); payload = uint64_t(value) + 1; break; }
            case 0xc8: { uint16_t value = 0; if (!big_endian(value)) return fail("truncated ext16"); payload = uint64_t(value) + 1; break; }
            case 0xc9: { uint32_t value = 0; if (!big_endian(value)) return fail("truncated ext32"); payload = uint64_t(value) + 1; break; }
            case 0xca: payload = 4; break;
            case 0xcb: payload = 8; break;
            case 0xcc: case 0xd0: payload = 1; break;
            case 0xcd: case 0xd1: payload = 2; break;
            case 0xce: case 0xd2: payload = 4; break;
            case 0xcf: case 0xd3: payload = 8; break;
            case 0xd4: payload = 2; break;
            case 0xd5: payload = 3; break;
            case 0xd6: payload = 5; break;
            case 0xd7: payload = 9; break;
            case 0xd8: payload = 17; break;
            case 0xc1: return fail("reserved MessagePack tag");
            default: return fail("unknown MessagePack tag");
        }
        return advance(payload);
    }

private:
    bool fail(const char * message) {
        if (error_.empty()) error_ = message;
        return false;
    }

    bool byte(uint8_t & value) {
        if (current_ == end_) return false;
        value = *current_++;
        return true;
    }

    bool peek(uint8_t & value) const {
        if (current_ == end_) return false;
        value = *current_;
        return true;
    }

    template <typename T>
    bool big_endian(T & value) {
        static_assert(std::numeric_limits<T>::is_integer, "integer required");
        if (sizeof(T) > static_cast<size_t>(end_ - current_)) return false;
        value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value = static_cast<T>((value << 8) | current_[index]);
        }
        current_ += sizeof(T);
        return true;
    }

    bool advance(uint64_t count) {
        if (count > static_cast<uint64_t>(end_ - current_)) return fail("truncated payload");
        current_ += count;
        return true;
    }

    bool check_container(uint64_t count, bool map) {
        if (count > MAX_CONTAINER_ITEMS || (map && count > MAX_CONTAINER_ITEMS / 2)) {
            return fail("MessagePack container limit exceeded");
        }
        return true;
    }

    const uint8_t * current_;
    const uint8_t * end_;
    std::string error_;
};

template <typename T>
static bool set_once(bool & seen, T & destination, T value, const char * field, std::string & error) {
    if (seen) {
        error = std::string("duplicate metadata field ") + field;
        return false;
    }
    seen = true;
    destination = std::move(value);
    return true;
}

static bool parse_argument(
        msgpack_reader & reader, flagos_amd::hsaco_argument_metadata & argument,
        std::string & error) {
    uint64_t fields = 0;
    if (!reader.read_map_size(fields)) {
        error = reader.error();
        return false;
    }
    bool have_offset = false;
    bool have_size = false;
    bool have_value_kind = false;
    bool have_address_space = false;
    for (uint64_t index = 0; index < fields; ++index) {
        std::string key;
        if (!reader.read_string(key)) {
            error = reader.error();
            return false;
        }
        if (key == ".offset" || key == ".size") {
            uint64_t value = 0;
            if (!reader.read_unsigned(value)) {
                error = reader.error();
                return false;
            }
            bool & seen = key == ".offset" ? have_offset : have_size;
            uint64_t & destination = key == ".offset" ? argument.offset : argument.size;
            if (!set_once(seen, destination, value, key.c_str(), error)) return false;
        } else if (key == ".value_kind" || key == ".address_space") {
            std::string value;
            if (!reader.read_string(value)) {
                error = reader.error();
                return false;
            }
            bool & seen = key == ".value_kind" ? have_value_kind : have_address_space;
            std::string & destination = key == ".value_kind" ? argument.value_kind : argument.address_space;
            if (!set_once(seen, destination, std::move(value), key.c_str(), error)) return false;
        } else if (!reader.skip()) {
            error = reader.error();
            return false;
        }
    }
    if (!have_offset || !have_size || !have_value_kind || argument.size == 0) {
        error = "incomplete kernel argument metadata";
        return false;
    }
    if (argument.value_kind == "global_buffer") {
        if (!have_address_space || argument.address_space != "global") {
            error = "invalid global-buffer argument metadata";
            return false;
        }
    } else if (argument.value_kind != "by_value" || have_address_space) {
        error = "unsupported kernel argument metadata";
        return false;
    }
    return true;
}

static bool parse_kernel(
        msgpack_reader & reader, flagos_amd::hsaco_kernel_metadata & metadata,
        std::string & error) {
    uint64_t fields = 0;
    if (!reader.read_map_size(fields)) {
        error = reader.error();
        return false;
    }
    bool have_args = false;
    bool have_name = false;
    bool have_symbol = false;
    bool have_kernarg = false;
    bool have_workgroup = false;
    bool have_wavefront = false;
    bool have_group = false;
    bool have_private = false;
    for (uint64_t index = 0; index < fields; ++index) {
        std::string key;
        if (!reader.read_string(key)) {
            error = reader.error();
            return false;
        }
        if (key == ".args") {
            if (have_args) {
                error = "duplicate metadata field .args";
                return false;
            }
            have_args = true;
            uint64_t count = 0;
            if (!reader.read_array_size(count) || count > 4096) {
                error = reader.error().empty() ? "too many kernel arguments" : reader.error();
                return false;
            }
            metadata.arguments.resize(static_cast<size_t>(count));
            for (auto & argument : metadata.arguments) {
                if (!parse_argument(reader, argument, error)) return false;
            }
        } else if (key == ".name" || key == ".symbol") {
            std::string value;
            if (!reader.read_string(value)) {
                error = reader.error();
                return false;
            }
            bool & seen = key == ".name" ? have_name : have_symbol;
            std::string & destination = key == ".name" ? metadata.name : metadata.symbol;
            if (!set_once(seen, destination, std::move(value), key.c_str(), error)) return false;
        } else if (key == ".kernarg_segment_size" ||
                   key == ".max_flat_workgroup_size" ||
                   key == ".wavefront_size" ||
                   key == ".group_segment_fixed_size" ||
                   key == ".private_segment_fixed_size") {
            uint64_t value = 0;
            if (!reader.read_unsigned(value)) {
                error = reader.error();
                return false;
            }
            bool * seen = nullptr;
            uint64_t * destination = nullptr;
            if (key == ".kernarg_segment_size") { seen = &have_kernarg; destination = &metadata.kernarg_segment_size; }
            else if (key == ".max_flat_workgroup_size") { seen = &have_workgroup; destination = &metadata.max_flat_workgroup_size; }
            else if (key == ".wavefront_size") { seen = &have_wavefront; destination = &metadata.wavefront_size; }
            else if (key == ".group_segment_fixed_size") { seen = &have_group; destination = &metadata.group_segment_fixed_size; }
            else { seen = &have_private; destination = &metadata.private_segment_fixed_size; }
            if (!set_once(*seen, *destination, value, key.c_str(), error)) return false;
        } else if (!reader.skip()) {
            error = reader.error();
            return false;
        }
    }
    if (!have_args || !have_name || !have_symbol || !have_kernarg || !have_workgroup ||
        !have_wavefront || !have_group || !have_private || metadata.name.empty() ||
        metadata.symbol.empty() || metadata.kernarg_segment_size == 0 ||
        metadata.max_flat_workgroup_size == 0 || metadata.wavefront_size == 0) {
        error = "incomplete AMDGPU kernel metadata";
        return false;
    }
    uint64_t previous_end = 0;
    for (const auto & argument : metadata.arguments) {
        if (argument.offset < previous_end || argument.offset > metadata.kernarg_segment_size ||
            argument.size > metadata.kernarg_segment_size - argument.offset) {
            error = "invalid kernarg layout";
            return false;
        }
        previous_end = argument.offset + argument.size;
    }
    // Triton always appends global- and profiling-scratch pointers to the
    // explicit function ABI, even when both requested sizes are zero.
    if (metadata.arguments.size() < 2) {
        error = "missing Triton scratch arguments";
        return false;
    }
    for (size_t index = metadata.arguments.size() - 2; index < metadata.arguments.size(); ++index) {
        const auto & argument = metadata.arguments[index];
        if (argument.size != 8 || argument.value_kind != "global_buffer" ||
            argument.address_space != "global") {
            error = "invalid Triton scratch argument ABI";
            return false;
        }
    }
    return true;
}

static bool parse_amdgpu_metadata(
        const uint8_t * descriptor, size_t size,
        flagos_amd::hsaco_kernel_metadata & metadata, std::string & error) {
    msgpack_reader reader(descriptor, size);
    uint64_t fields = 0;
    if (!reader.read_map_size(fields)) {
        error = reader.error();
        return false;
    }
    bool have_target = false;
    bool have_kernels = false;
    bool have_version = false;
    for (uint64_t index = 0; index < fields; ++index) {
        std::string key;
        if (!reader.read_string(key)) {
            error = reader.error();
            return false;
        }
        if (key == "amdhsa.target") {
            std::string target;
            if (!reader.read_string(target) ||
                !set_once(have_target, metadata.target, std::move(target), key.c_str(), error)) {
                if (error.empty()) error = reader.error();
                return false;
            }
        } else if (key == "amdhsa.version") {
            if (have_version) {
                error = "duplicate metadata field amdhsa.version";
                return false;
            }
            have_version = true;
            uint64_t count = 0;
            uint64_t major = 0;
            uint64_t minor = 0;
            if (!reader.read_array_size(count) || count != 2 ||
                !reader.read_unsigned(major) || !reader.read_unsigned(minor) ||
                major != 1 || minor != 2) {
                error = reader.error().empty()
                    ? "unsupported AMDGPU metadata version" : reader.error();
                return false;
            }
        } else if (key == "amdhsa.kernels") {
            if (have_kernels) {
                error = "duplicate metadata field amdhsa.kernels";
                return false;
            }
            have_kernels = true;
            uint64_t count = 0;
            if (!reader.read_array_size(count) || count != 1) {
                error = reader.error().empty() ? "HSACO must contain exactly one kernel" : reader.error();
                return false;
            }
            if (!parse_kernel(reader, metadata, error)) return false;
        } else if (!reader.skip()) {
            error = reader.error();
            return false;
        }
    }
    if (!reader.complete()) {
        error = "trailing bytes in AMDGPU metadata";
        return false;
    }
    if (!have_target || !have_kernels || !have_version || metadata.target.empty()) {
        error = "incomplete AMDGPU code-object metadata";
        return false;
    }
    return true;
}

} // namespace

namespace flagos_amd {

bool parse_hsaco_metadata(
        const std::vector<uint8_t> & image,
        hsaco_kernel_metadata & metadata,
        std::string & error) {
    metadata = {};
    error.clear();
    if (image.size() < ELF64_HEADER_SIZE ||
        std::memcmp(image.data(), "\x7f" "ELF", 4) != 0 ||
        image[4] != ELFCLASS64 || image[5] != ELFDATA2LSB || image[6] != 1 ||
        image[7] != ELFOSABI_AMDGPU_HSA) {
        error = "not an ELF64 little-endian AMD HSA code object";
        return false;
    }
    uint16_t type = 0;
    uint16_t machine = 0;
    uint32_t version = 0;
    uint64_t program_offset = 0;
    uint16_t header_size = 0;
    uint16_t program_entry_size = 0;
    uint16_t program_count = 0;
    if (!read_le(image, 16, type) || !read_le(image, 18, machine) ||
        !read_le(image, 20, version) || !read_le(image, 32, program_offset) ||
        !read_le(image, 52, header_size) || !read_le(image, 54, program_entry_size) ||
        !read_le(image, 56, program_count) ||
        type != ET_DYN || machine != EM_AMDGPU || version != 1 ||
        header_size != ELF64_HEADER_SIZE || program_offset < ELF64_HEADER_SIZE ||
        program_entry_size != ELF64_PROGRAM_HEADER_SIZE || program_count == 0 ||
        program_count == 0xffff || program_count > 1024 ||
        program_offset > std::numeric_limits<size_t>::max() ||
        !bounded_region(static_cast<size_t>(program_offset),
            uint64_t(program_entry_size) * program_count, image.size())) {
        error = "invalid AMDGPU ELF program-header table";
        return false;
    }

    bool found_metadata = false;
    for (uint16_t index = 0; index < program_count; ++index) {
        const size_t header = static_cast<size_t>(program_offset) + size_t(index) * program_entry_size;
        uint32_t segment_type = 0;
        uint64_t segment_offset = 0;
        uint64_t segment_size = 0;
        if (!read_le(image, header, segment_type) || !read_le(image, header + 8, segment_offset) ||
            !read_le(image, header + 32, segment_size)) {
            error = "truncated AMDGPU program header";
            return false;
        }
        if (segment_type != PT_NOTE) continue;
        if (segment_offset > std::numeric_limits<size_t>::max() ||
            !bounded_region(static_cast<size_t>(segment_offset), segment_size, image.size())) {
            error = "invalid PT_NOTE segment";
            return false;
        }
        size_t note = static_cast<size_t>(segment_offset);
        const size_t note_end = note + static_cast<size_t>(segment_size);
        while (note < note_end) {
            uint32_t name_size = 0;
            uint32_t descriptor_size = 0;
            uint32_t note_type = 0;
            if (note_end - note < 12 || !read_le(image, note, name_size) ||
                !read_le(image, note + 4, descriptor_size) || !read_le(image, note + 8, note_type)) {
                error = "truncated ELF note header";
                return false;
            }
            note += 12;
            size_t padded_name = 0;
            size_t padded_descriptor = 0;
            if (!align4(name_size, padded_name) || !align4(descriptor_size, padded_descriptor) ||
                padded_name > note_end - note || padded_descriptor > note_end - note - padded_name) {
                error = "invalid ELF note size";
                return false;
            }
            const uint8_t * name = image.data() + note;
            const uint8_t * descriptor = image.data() + note + padded_name;
            const bool is_amdgpu = name_size == 7 && std::memcmp(name, "AMDGPU\0", 7) == 0;
            if (is_amdgpu && note_type == NT_AMDGPU_METADATA) {
                if (found_metadata || descriptor_size == 0 || descriptor_size > MAX_METADATA_NOTE_BYTES) {
                    error = found_metadata ? "duplicate AMDGPU metadata note" : "invalid AMDGPU metadata note size";
                    return false;
                }
                found_metadata = true;
                if (!parse_amdgpu_metadata(descriptor, descriptor_size, metadata, error)) {
                    return false;
                }
            }
            note += padded_name + padded_descriptor;
        }
    }
    if (!found_metadata) {
        error = "AMDGPU metadata note not found";
        return false;
    }
    return true;
}

} // namespace flagos_amd
