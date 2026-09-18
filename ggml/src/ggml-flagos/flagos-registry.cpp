#include "ggml-flagos.h"

#include "../ggml-backend-impl.h"

static const char * flagos_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FlagOS";
}

static size_t flagos_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 0;
}

static ggml_backend_dev_t flagos_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_UNUSED(index);
    return nullptr;
}

ggml_backend_reg_t ggml_backend_flagos_reg(void) {
    static ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        {
            flagos_reg_get_name,
            flagos_reg_get_device_count,
            flagos_reg_get_device,
            nullptr,
        },
        nullptr,
    };
    return &reg;
}

#ifdef GGML_BACKEND_DL
static int flagos_reg_score() {
    return 0;
}

GGML_BACKEND_DL_SCORE_IMPL(flagos_reg_score)
#endif

GGML_BACKEND_DL_IMPL(ggml_backend_flagos_reg)
