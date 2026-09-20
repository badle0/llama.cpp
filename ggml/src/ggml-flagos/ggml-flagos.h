#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "flagos-registry.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_flagos_reg(void);

GGML_BACKEND_API ggml_backend_t ggml_backend_flagos_init(int device);
GGML_BACKEND_API bool ggml_backend_is_flagos(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_flagos_set_device(int device);
GGML_BACKEND_API int ggml_backend_flagos_get_device(void);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_flagos_buffer_type(int device);
GGML_BACKEND_API void ggml_backend_flagos_reg_devices(void);

#ifdef __cplusplus
}
#endif
