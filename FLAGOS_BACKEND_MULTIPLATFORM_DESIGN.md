# ggml-flagos 多平台 Backend 与图融合完整方案

状态：评审修订稿 v0.3（含 Denglin Q4 DLBLAS 实测与 NVIDIA/AMD 参考 Provider）
基线日期：2026-08-22
适用范围：llama.cpp / GGML、FlagOS、Denglin GPU、NVIDIA CUDA GPU、AMD HIP GPU/APU、SpacemiT K3、ARM CPU Reference Provider

## 1. 执行摘要

FlagOS 不改变 llama.cpp 的 GGUF、KV Cache、模型建图和 Scheduler，而是在 GGML Backend 边界建立统一的多平台执行层。Denglin GPU、NVIDIA CUDA GPU、AMD HIP GPU/APU、SpacemiT K3 和 ARM CPU 都在同一个 `ggml-flagos` Backend 下以 Provider 方式接入。

目标架构不是把所有芯片伪装成 CUDA，也不是给每款芯片复制一份 Backend，而是共享以下能力：

- GGML 算子语义规范化；
- AOT Kernel Package 与 ABI 校验；
- `supports_op` 和实际执行共用的 Selector；
- 图分析、融合 Pattern、Execution Plan 和 Plan Cache；
- Buffer 生命周期、Fallback、诊断和测试框架。

不同 Provider 保留各自的执行语义：

- Denglin 使用 CUDA-compatible Runtime，NVIDIA 使用原生 CUDA，AMD 使用 HIP/ROCm；三者各自实现设备内存或统一内存、Queue/Stream、Event、AOT Module、厂商库和 Native Graph；
- SpacemiT K3 作为正式 AI 芯片 Provider，由 FlagOS 与进迭时空共建，负责其设备、内存、AI Core 调度、AOT Launcher、局部存储和静态执行计划；
- ARM CPU 是 FlagOS Reference Provider，主要用于开发、Demo、AOT ABI 验证、融合正确性验证和跨平台展示，使用 Host/Coherent Memory、持久线程池和 CPU AOT 宏内核。

算子覆盖完成以后，下一阶段性能核心是 Provider-aware 图级优化。公共 Fusion Planner 负责识别合法且有收益的融合区域，Provider 负责判断能否 Lower 并选择本芯片的实现。CUDA Graph 只是 GPU Execution Plan 的一种实现，不是公共图抽象本身。

## 2. 已确认的架构决策

1. 只有一个 GGML Backend 家族：`ggml-flagos`。
2. SpacemiT 不单独建设 `ggml-spacemit`，而是在 `ggml-flagos/providers/spacemit` 下与 FlagOS 共建。
3. K3 按正式 AI 芯片 Provider 建模，不走 `ggml-cpu` Backend 路径。
4. ARM CPU 作为 FlagOS Reference Provider 接入 `ggml-flagos`，而不是本项目的性能目标芯片。
5. 公共层共享算子语义、图融合、计划、AOT 元数据和测试，不共享 CUDA 类型和 GPU Launch 参数。
6. `supports_op` 和执行选择必须来自同一 Selector，避免能力声明与 Launcher 不一致。
7. 图融合在 Plan 阶段完成；Graph Capture/Command Buffer/CPU Plan Cache 是 Provider 侧的执行计划实现。
8. 首版使用编译期 Provider 注册，不立即引入第二层 `dlopen` 插件系统；接口从一开始保持 C-compatible，后续有独立交付需求时再固化外部 ABI。
9. Common IR 是后续增强路径，不是新芯片接入和首批图融合的前置条件。
10. `ggml_backend_dev_type` 按 llama.cpp 的调度语义选择，不用于描述物理硬件：正式承载模型层的 K3 返回 `GPU`；ARM Reference 返回 `ACCEL`，绝不注册为 `CPU`。
11. Plan Cache 以 Canonical Structural Fingerprint 为主键；Graph UID 只作为“本次仍是同一个已分配图”的快速路径。
12. NVIDIA 与 AMD 是两个独立 Provider，不在 Denglin Launcher 中增加厂商分支：RTX 3080 作为 `sm_86` CUDA Reference Target；Ryzen AI 9 HX 370 的 Radeon 890M 作为 `gfx1150` HIP/APU Reference Target。

## 3. 目标与非目标

### 3.1 目标

- 在不改变 llama.cpp 上层模型生态的情况下接入多类 FlagOS 芯片；
- 复用 FlagTree/TLE 生成的 AOT Kernel 和厂商高性能库；
- 支持逐算子、融合宏内核、静态执行计划和 Native Graph；
- 让新增 GPU（包括 NVIDIA 与 AMD）只实现 Provider、AOT Package 和厂商库适配，不复制 GGML Backend；
- 让 SpacemiT 团队在稳定边界内独立推进 K3 设备和性能实现；
- 让 ARM 开发机能够运行同一套 Backend、模型、Selector、Fusion Planner 和测试；
- 保证不支持的节点由 GGML Scheduler 正确分配给 CPU Backend；
- 用端到端指标而不只是 Kernel Microbenchmark 验收性能。

### 3.2 非目标

- 首版不建设完整新推理引擎；
- 首版不要求所有 Provider 支持相同算子和相同图能力；
- 不在公共接口暴露 `cudaStream_t`、`CUmodule`、Grid/Block 等 CUDA 概念；
- 不通过 Python/PyTorch 桥接执行 Triton Kernel；当前纯 C++ AOT Launcher 已经证明更合适；
- 不承诺 ARM Reference Provider 与目标 GPU/K3 达到相同性能；
- 不在 Runtime Launch 失败后偷偷把设备节点搬回 CPU；Fallback 必须在 Scheduler 划图前决定。

## 4. 当前代码基线

当前 `ggml-flagos` 已经在 Denglin KS20-A 和 Qwen2.5-1.5B 上打通端到端：

- 标准 GGML Backend 注册、Device、Buffer、Stream 和 Event；
- 34 个 Triton AOT Kernel；
- 合并模块一次加载、按符号解析，单 Kernel 模块作为 Fallback；
- Manifest 驱动文件名、符号、Block、Warp 和 Shared Memory；
- C++ Driver Launcher 直接调用 `cuLaunchKernel`；
- Q4_K/Q6_K 直接 Kernel、反量化到 F16 后调用 Denglin BLAS 的路径，以及实验性的 Q4_K 重排后直接调用 DLBLAS INT4 group-quant GEMV 路径；
- Weight Cache、Dequant Cache 和 Activation Cast 复用；
- `RMS_NORM + MUL` 单 Launch、多输出融合，norm 和 mul 两个输出均物化；
- CUDA-compatible Graph 的 warmup、capture、instantiate、replay、失效和 16 项 LRU Cache；
- 严格的 dtype、shape、layout、stride 和目标限制检查；
- `test-backend-ops` 中 392 个已声明支持用例通过；
- Qwen2.5-1.5B Q4_K_M 已完成短上下文、长上下文、CLI 和 Server 验证。

现有测量基线：

| 场景 | CPU | FlagOS Denglin |
| --- | ---: | ---: |
| PP128 | 24.07 token/s | 503.79 token/s |
| TG32 | 16.97 token/s | 17.28 token/s |
| Server 128-token warm TTFT | 5333.11 ms | 276.48 ms |

这些结果证明了 Backend 端到端路径。2026-08-22 的 Provider Core 重构已经完成第一阶段边界：

- 根目录新增版本化 `flagos_provider_v1`、统一 Registry、Provider/Device Identity、Memory Domain 和 Capability；
- 公共 Registry/Provider ABI/Graph Plan 中已无 CUDA、HIP 或 DLBLAS 类型；
- 原 3100 余行 Denglin 实现已物理迁入 `providers/denglin/flagos-denglin.cpp`；
- Denglin SDK、libcurt、DLBLAS 和 cubin 检查只在 `GGML_FLAGOS_DENGLIN=ON` 时启用；关闭 Denglin 后，公共 Core 可在无 SDK 环境独立配置、编译和运行 ABI/Graph Plan 测试；
- 单一 FlagOS Registry 可以聚合多个编译期 Provider，并把全局 Device Index 映射到 Provider Local Index；
- 公共 Registry 会拒绝重复 Provider ID、零 UUID、同 Provider 重复 Device UUID 和不完整的 Capability；`get_device()` 只校验 FlagOS 已记录的当前 Provider，不会在 CUDA/HIP 各自都有 local device 0 时跨 Provider 猜测；
- Provider Probe 失败相互隔离，跨 Provider Backend 不会进入 Denglin D2D Copy；
- 每个厂商的 SDK、Library 和 AOT Package 检查已经迁入各自的 `providers/<vendor>/provider.cmake`，公共 CMake 不再包含 Denglin 或 AMD 分支细节；
- 新增无厂商 SDK 的双 Mock Provider Registry 测试，验证 Denglin 两设备与 AMD 一设备的聚合顺序、全局/本地序号转换和当前设备歧义处理；
- Denglin 迁移后仍通过 392/392 Backend 测试和 Qwen3-4B CUDA Graph 端到端验证。

仍未完成的边界包括：

- AMD HIP Provider 还没有 Runtime/Memory/Launcher 实现；当前 `GGML_FLAGOS_AMD=ON` fail closed，避免暴露假设备；
- `supports_op` 与执行 `switch` 分别维护规则；
- Graph Cache 和执行上下文直接持有 CUDA 类型；
- Kernel Registry 仍是固定的 C++ 成员；
- `graph_plan_create/free/update/compute` 尚未实现；
- Native Graph、Kernel Registry、BLAS 和 Selector 仍是各 Provider 内部实现，尚未形成完整 Typed Binding/Manifest v3。

历史文档中的 Python/C API + PyTorch/Triton 路线已经过时。正式路线是：离线 AOT 编译，C/C++ Runtime 直接加载和启动。

### 4.1 当前性能判断

当前 PP128 相比 CPU 已有显著提升，但 TG32 基本持平。这说明现阶段不能把“CUDA Graph 已经 Replay”直接等价为“Decode 已经优化完成”。下一步必须同时测量并区分：

- 量化 GEMV/Weight Bandwidth；
- 每 Token Kernel/Barrier 数；
- Attention 和 KV Cache 访问；
- CPU Scheduler 与设备 Launch 开销；
- Graph Capture 的实际覆盖率；
- Fallback 节点和跨 Backend Copy。

图融合是下一阶段核心能力之一，但 Direct GEMV/GEMM、Attention、Weight Layout 和内存带宽仍是性能基础。优先级应由端到端 Profile 决定，不能用 Pattern 数量代替收益。

现有 `RMS_NORM + MUL` 只能证明同一 Kernel 内完成两节点计算并减少 Launch：Kernel 同时写 `norm_output` 和 `mul_output`。它省去 MUL 对 norm 中间结果的一次设备内存读取，但没有消除 norm 的写入或 GGML 中间 Tensor 分配，不能作为“完整带宽融合已经完成”的证据。

2026-08-22 的 M0.5 检查点新增了真实 GGUF Q4_K 到 DLBLAS INT4 group-quant 的一次性重排与缓存：scale/zero 按 `[K/32, output_rows]` 排列，DLBLAS F16 输出再由 AOT Kernel 转回 GGML F32。Qwen3-4B Q4_K_M 端到端输出正确，额外缓存 216 个权重、1,961,164,800 bytes。相同 TG16 测试中，Direct Q4_K AOT + Graph 为 4.74 token/s，DLBLAS 无 Graph 为 10.03 token/s，DLBLAS + Graph 为 10.75 token/s。它证明厂商量化库和 Native Graph 可以叠加获益，但 no-warmup 首轮只有 1.21 token/s，说明厂商 JIT 冷启动尚未达到默认路径门槛；正式发布必须将该内核 AOT 化、提供持久 JIT Cache，或在模型装载阶段完成可观测预热。

## 5. 总体架构

```text
llama.cpp
  GGUF / KV Cache / GGML Graph
                |
                v
        GGML Backend Scheduler
      supports_op + supports_buft
                |
                v
          ggml-flagos Core
  +----------------------------------+
  | Device Registry                  |
  | Op Signature + Selector          |
  | Kernel Catalog + AOT Package     |
  | Graph Analyzer + Fusion Planner  |
  | Execution Plan + Plan Cache      |
  | Memory/Liveness Plan             |
  | Diagnostics + Conformance Tests  |
  +----------------+-----------------+
                   |
       Provider Contract
                   |
  +-----------+-----------+-----------+-----------+-----------+
  |           |           |           |           |           |
Denglin     NVIDIA      AMD HIP    SpacemiT K3   ARM Ref    Future
GGML GPU    GGML GPU    GGML GPU   GGML GPU      GGML ACCEL Provider
  |           |           |           |           |
AOT+DLBLAS  AOT+cuBLAS  AOT+hipBLAS AOT+AI Core ELF/AOT
CUDA Graph  CUDA Graph  HIP Graph   Static Plan Thread Plan
                                    TCM/DMA     Plan Cache
```

Provider 对 GGML 不可见。GGML 只看到同一注册表枚举出的多个 FlagOS Device：

```text
FlagOS:Denglin:0
FlagOS:NVIDIA:0
FlagOS:AMD:0
FlagOS:SpacemiT:0
FlagOS:ARM:0
```

这是 M2 起采用的正式命名，不在 M1 “行为不变”重构中修改。M1 继续暴露当前 Denglin 名称 `FlagOS0`，保证 `-dev`、验证脚本和部署配置不变。当前 GGML 的 `dev_by_name` 是精确匹配且没有 Device Alias 接口，因此不虚构兼容 alias；M2 改名时必须作为显式迁移完成，同批更新仓库脚本、配置模板、文档和 CI，并在 release note 中列出旧名到新名的映射。如必须长期兼容旧名，应先单独扩展名称解析协议，而不是注册重复设备。

`ggml_backend_dev_type` 是 llama.cpp 的调度策略输入，不是硬件分类字段。类型选择固定为：

- Denglin/NVIDIA/AMD：`GPU`；
- SpacemiT K3：`GPU`，确保进入 `model->devices` 并能承载模型层和 KV Cache；不默认用 `IGPU`，因为有独立 GPU 时 llama.cpp 会忽略 IGPU；
- ARM Reference：`ACCEL`，确保不被 `ggml_backend_dev_by_type(CPU)` 选为系统 CPU Backend；
- 物理硬件类别放在 `flagos_device_caps.kind`，由 FlagOS 自己解释。

ARM Provider 作为 `ACCEL` 通过 CPU Buffer Type 候选列表机会性接管已覆盖算子，但不能替代原生 `ggml-cpu`。K3 即使是集成式 AI 芯片，也按正式模型设备的调度语义返回 `GPU`；这是调度适配，不表示其内部使用 GPU/CUDA 执行模型。

按整数序号初始化的旧 API 只能作为兼容入口。正式选择应支持稳定的 `provider_name + device_uuid`，否则启用/关闭某个 Provider 后 Device Ordinal 会变化，部署配置可能静默选错设备。

## 6. Provider 身份和能力模型

### 6.1 设备身份

不能只使用整数 Device ID。`Denglin:0`、`NVIDIA:0` 和 `AMD:0` 不是同一设备，也不能因为 ordinal 相同就假定可以 D2D Copy。

```cpp
struct flagos_provider_identity {
    uint64_t provider_id;
    const char * provider_name;
    uint32_t provider_version;
};

struct flagos_device_identity {
    flagos_provider_identity provider;
    uint64_t device_uuid_hi;
    uint64_t device_uuid_lo;
    uint64_t memory_domain_id;
    uint32_t ordinal;
};
```

Backend GUID 可以继续用于判断“是否属于 FlagOS Backend 家族”，但 Copy、Module、Event 和 Plan 兼容性必须检查 Provider、Device UUID 和 Memory Domain。

### 6.2 能力分组

能力必须是运行时数据，而不是不断增加编译宏：

```cpp
struct flagos_device_caps {
    flagos_device_kind kind;
    flagos_memory_caps memory;
    flagos_queue_caps queue;
    flagos_graph_caps graph;
    flagos_cpu_caps cpu;
    flagos_local_memory_caps local_memory;
};
```

关键能力包括：

- Device/Host/Unified/Coherent/Local Scratch 内存；
- Host Pointer Import 和 Buffer 是否 CPU 可访问；
- Async Copy、Queue、Event、Peer Copy；
- Native Capture、Graph Update、Command Plan；
- 支持的数据类型、量化格式、布局、对齐和最大维度；
- ARM ISA、向量长度、线程拓扑和 Cache；
- K3 AI Core、TCM/Local Memory、DMA 和执行队列。

不具备的能力用空扩展或 `false` 表示，不能由 Provider 伪造。

## 7. Provider Contract

首版 Provider 与 Core 一起编译，但接口使用 C-compatible Function Table 和不透明 Handle，避免以后 ABI 重构。

```cpp
struct flagos_provider_v1 {
    uint32_t api_version;
    uint32_t struct_size;
    flagos_provider_identity identity;

    const flagos_device_ops * device;
    const flagos_memory_ops * memory;
    const flagos_execution_ops * execution;
    const flagos_kernel_catalog_ops * kernels;
    const flagos_selector_ops * selector;

    const flagos_queue_ops * queues;
    const flagos_event_ops * events;
    const flagos_native_graph_ops * native_graph;
    const flagos_command_plan_ops * command_plan;
    const flagos_library_ops * libraries;
    const flagos_local_memory_ops * local_memory;
    const flagos_telemetry_ops * telemetry;
};
```

前三组是必选基础能力，后续扩展可以为空。

首版由 CMake 条件编译后静态注册：

```cpp
flagos_register_denglin_provider();
flagos_register_newgpu_provider();
flagos_register_spacemit_provider();
flagos_register_arm_provider();
```

Provider 探测必须彼此隔离：每个 `probe()` 返回可诊断的状态和原因；某个 Provider SDK 缺失、驱动不可用或没有设备时，仅将该 Provider 记为零设备，不得阻断其他 Provider 注册。只有部署配置显式声明某 Provider 为 required 时，其探测失败才是初始化错误。全部加速器探测失败也是合法状态，ARM Reference 仍应能在无卡环境单独运行。

Core 在注册结束、交给 Scheduler 之前检查总 Backend 数，必须为原生 CPU Backend 和其他 Backend 预留名额，不得等到 `GGML_SCHED_MAX_BACKENDS == 16` 的裸断言才失败。每个 Provider 支持配置最大枚举设备数，超限时给出 Provider 名、发现数、保留数和 Scheduler 上限。

只有当出现“Provider 独立发布、同一程序动态加载多个厂商 SDK”的明确需求后，才增加 `dlopen` Provider Loader。届时沿用同一 Function Table。

## 8. 内存抽象

### 8.1 Memory Domain

```text
HOST              普通CPU内存
HOST_PINNED       可供异步设备访问的Host内存
DEVICE_LOCAL      离散加速器本地内存
UNIFIED_COHERENT  CPU/AI Core共享且一致的地址空间
LOCAL_SCRATCH     TCM/SRAM/VTCM等短生命周期局部存储
```

`LOCAL_SCRATCH` 不能作为长期 Tensor Buffer 使用，只能由 Execution Plan 分配为 Workspace。

### 8.2 Allocation Identity

Graph Replay、Weight Cache 和跨 Provider Copy 不能只依赖裸指针：

```cpp
struct flagos_allocation {
    flagos_device_identity device;
    uint64_t allocation_id;
    uint64_t generation;
    void * address;
    size_t size;
};
```

Buffer 修改时更新 `generation`。Weight Cache Key 至少包含：

```text
provider + device + allocation_id + offset + generation + dtype/layout
```

### 8.3 Copy 规则

- 同 Provider、同 Device：本地 Copy；
- 同 Provider、不同 Device：只有 `peer_copy` 声明支持才允许；
- 不同 Provider：通过 Host/Staging，除非双方显式注册互操作路径；
- ARM Host Buffer：优先零拷贝或 Host Pointer Import；
- SpacemiT：由 Provider 根据真实 SDK 声明 Coherent、Device Local 或 Local Scratch，Core 不预设其内存语义。

现有仅通过 Buffer Type vtable 函数指针判断“属于 FlagOS”，再比较整数 Device ID 的实现必须在 M1 删除。所有 Copy 路径先比较 `provider_id + device_uuid + memory_domain_id`：只有身份和内存域满足本地/Peer 规则时才能调用 Provider Copy；否则返回不支持，由 Scheduler 选择 Host/Staging 路径。尤其禁止把 `Denglin:0` 与 `ARM:0` 因 ordinal 同为 0 而误判成 D2D。

## 9. Op Signature、Kernel Catalog 和 Selector

### 9.1 Canonical Op Signature

```cpp
struct flagos_tensor_desc {
    ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    flagos_layout layout;
    flagos_memory_domain domain;
};

struct flagos_op_signature {
    ggml_op op;
    flagos_tensor_desc dst;
    flagos_tensor_desc src[GGML_MAX_SRC];
    flagos_semantic_params params;
};
```

Signature 不包含实际地址和 Tensor Identity。地址属于执行 Binding，不属于算子语义。`op_params` 也不能把整个原始字节区直接当作跨平台 Cache Key；Canonicalizer 必须按具体 `ggml_op` 解码有语义的字段，并区分：

- 影响 Kernel 选择和 Plan 结构的静态参数；
- 只需要在执行前重新绑定的动态 Scalar；
- 未使用的 Padding 字节。

这样可以避免 Padding 或动态值造成无意义的 Plan Cache Miss，也避免动态参数变化后错误复用旧 Plan。

### 9.2 Execution Choice

```cpp
enum flagos_execution_kind {
    FLAGOS_DIRECT_KERNEL,
    FLAGOS_VENDOR_LIBRARY,
    FLAGOS_FUSED_KERNEL,
    FLAGOS_COMPILED_REGION,
};

struct flagos_execution_choice {
    flagos_execution_kind kind;
    uint64_t implementation_id;
    uint64_t kernel_abi_hash;
    size_t workspace_size;
    bool capture_safe;
    flagos_cost estimate;
};
```

设备探测阶段加载 Kernel Catalog 元数据。`supports_op(dev, op)` 调用 Selector，但不启动 Kernel；Backend 初始化阶段加载二进制。如果 AOT Package 无法加载或 ABI 不匹配，设备初始化失败，而不是先声明支持、执行时再失败。

### 9.3 单一事实源

```text
supports_op
    -> canonicalize
    -> selector.select
    -> 是否存在ExecutionChoice

graph execution
    -> 使用同一Signature和Selector结果
    -> bind addresses
    -> execute
```

每个 Provider 可以有不同 Variant，但不能复制公共的 GGML Shape 解析代码。

## 10. AOT Kernel Package 和 Launcher ABI

### 10.1 Manifest v3

```json
{
  "format": 3,
  "provider": "spacemit",
  "target": "k3-aicore-v1",
  "package_kind": "vendor_aot_module",
  "runtime_abi": "flagos-provider-v1",
  "kernel_abi_hash": "...",
  "compiler": {
    "name": "flag-tree",
    "version": "..."
  },
  "kernels": [
    {
      "semantic_op": "rms_norm_mul",
      "variant": "f32-default",
      "symbol": "...",
      "argument_schema_id": "rms_norm_mul_v1",
      "required_caps": [],
      "dtype_constraints": [],
      "shape_constraints": {},
      "workspace": 0,
      "capture_safe": true
    }
  ]
}
```

`package_kind` 至少支持：

- GPU vendor module/cubin；
- ARM ELF shared object/static object；
- K3 vendor AOT module；
- 后续 Common IR compiled region package。

### 10.2 参数 ABI

Manifest 负责描述和校验，不能在每 Token 热路径动态解析 JSON 并拼装参数。构建阶段生成 Typed Binding：

```cpp
struct flagos_kernel_binding {
    uint64_t schema_id;
    flagos_status (*bind)(const flagos_plan_node *, flagos_argument_pack *);
};
```

GPU Binding 生成设备指针和 Scalar 参数，Provider 内部再计算 Grid/Block；ARM Binding 生成直接函数调用参数和 Parallel Policy；K3 Binding 生成设备描述符、Local Memory 和 AI Core 调度参数。

### 10.3 Package 搜索路径

按顺序搜索：

1. `FLAGOS_KERNEL_PATH`；
2. 动态库相对路径 `../share/ggml/flagos/<provider>/<target>`；
3. FlagOS 系统安装路径；
4. 开发构建目录，仅用于未安装构建。

Manifest、二进制和 ABI Hash 必须原子版本化，禁止混用不同构建批次。

`dtype_constraints`、`shape_constraints`、Layout、Alignment 和 `required_caps` 是 Selector 的可执行输入，不是说明性元数据。AOT Variant 中由 `tl.constexpr` 或编译选项冻结的边界必须出现在 Manifest/生成代码里，并由 `supports_op -> Selector` 在划图前拒绝。每个 Variant 都要自动生成边界内、边界值和越界用例，防止从硬编码 `supports_op` 迁移时丢失约束。

## 11. 图融合和 Execution Plan

### 11.1 为什么这是下一阶段性能核心

GPU 侧主要收益：

- 减少 Kernel Launch；
- 合并内存带宽受限算子；
- 避免中间 Tensor；
- 扩大 Native Graph 的稳定执行单元。

ARM/K3 侧主要收益：

- 减少中间 Tensor 对主存的读写；
- 让 Tile 停留在寄存器、L1/L2 或 TCM；
- 减少每节点线程 Barrier；
- 复用输入激活和预打包 Weight；
- 将多个节点变成更适合 AI Core/向量核的宏内核。

图融合不能替代高质量 GEMM/GEMV、Attention 和量化 Kernel。正确顺序是：单算子覆盖和性能达标后，用图融合进一步减少数据移动和调度。

### 11.2 三层融合

第一层，Epilogue/Kernel Fusion：

```text
RMS_NORM + MUL
MATMUL + ADD
MATMUL + BIAS + SILU
SILU + MUL
```

第二层，Tile/Loop Fusion：多个算子仍然保持逻辑边界，但同一线程在一个 Tile 上连续执行，中间数据只存放在寄存器、局部 Scratch 或 Cache。

第三层，Macro Subgraph Fusion：

```text
QKV Projection
Q/K RoPE + KV Cache写入
FFN Gate/Up + SwiGLU
QK + Scale/Mask + Online Softmax + V
```

融合收益需要按输出行为分类：

- 单 Launch、多输出：合并调度并复用寄存器值，但仍按 GGML 语义物化所有输出；
- 中间结果消除：仅当合法性分析证明中间 Tensor 没有外部消费者、不是 Graph Output 且 View/Alias 安全时，才允许不写该输出；
- Region/Tile Fusion：逻辑上可保留多个 Plan Node，但通过寄存器、Cache 或 Local Scratch 避免主存往返。

当前 `RMS_NORM + MUL` 属于第一类：两个输出都写入。它减少一次 Launch，并省去 MUL 对 norm 结果的一次读取，但不减少 norm 结果的写入和 GGML Allocation。

### 11.3 Plan 构建流程

```text
GGML backend subgraph
        |
        v
Canonical graph
        |
        v
Dependency/Liveness/Alias analysis
        |
        v
Pattern candidate enumeration
        |
        v
Provider can_lower + cost estimate
        |
        v
Non-overlapping region selection
        |
        v
Provider lowering
        |
        v
Memory/Workspace plan
        |
        v
Executable Plan + Binding Template
```

首版 Pattern 数量有限，可以使用按收益优先的 Greedy 选择。Pattern 出现重叠和组合爆炸后，再升级为带 Cost 的 Region Selection。

### 11.4 融合合法性

优先复用 GGML 已有的：

- `ggml_can_fuse()`；
- `ggml_can_fuse_subgraph()` / `ggml_can_fuse_subgraph_ext()`。

并追加以下约束：

- 所有节点属于同一 FlagOS Provider 子图；
- 被消除的中间节点没有子图外消费者；
- Graph Output、View、Alias 和 In-place 语义安全；
- DType、Layout、Stride、Shape 满足 Provider 实现；
- 数值精度和运算顺序变化在允许范围内；
- Workspace、寄存器、Local Memory 和并行度满足限制；
- 使用厂商库时，该调用满足 Native Graph Capture 要求。

多输出融合必须在 Plan Node 中显式描述多个输出。不能为了融合丢失 GGML 要求物化的中间结果。

当前 `RMS_NORM + MUL` 因为仍物化 norm 输出，即使 norm 存在额外消费者也不会丢失结果。若迁移时新增“只写 mul 输出”的带宽优化 Variant，必须在同一变更中通过 `ggml_can_fuse_subgraph_ext()` 验证 use count、Graph Output、View/Alias 和候选节点闭合性；不得只复用当前的相邻节点特判。

### 11.5 Cost Model

首版成本模型：

```text
benefit = eliminated_read_bytes
        + eliminated_write_bytes
        + eliminated_launch_cost
        + eliminated_barrier_cost
        + input_reuse_benefit
        - extra_compute
        - parallelism_loss
        - workspace_cost
        - register_or_local_memory_pressure
```

阈值可以来自离线 Benchmark，按 Provider、模型阶段、Token 数和 Shape 选择。

必须允许“Pattern 合法但不融合”。过度融合可能破坏厂商 GEMM、降低并行度、增加寄存器压力和代码体积。

成本统计必须分别报告 eliminated read/write bytes。以当前多输出 `RMS_NORM + MUL` 为例，`eliminated_write_bytes = 0`，但 `eliminated_read_bytes` 包含 MUL 不再从设备内存读取 norm 中间结果的字节数；不能把它描述为完整中间 Tensor 消除。

### 11.6 Plan 数据结构

```cpp
struct flagos_plan_node {
    flagos_execution_kind kind;
    uint64_t implementation_id;
    flagos_tensor_binding * inputs;
    flagos_tensor_binding * outputs;
    flagos_parallel_policy parallel;
    size_t workspace_offset;
    size_t workspace_size;
    void * provider_executable;
};

struct flagos_execution_plan {
    flagos_device_identity device;
    uint64_t structural_fingerprint;
    uint64_t binding_fingerprint;
    flagos_plan_node * nodes;
    size_t workspace_size;
    void * native_plan;
};
```

### 11.7 接入 GGML Graph Plan API

当前 GGML 接口把 `graph_plan_*` 标记为“尚未使用”，主流 Scheduler 路径不能保证调用这些回调；同时 `graph_plan_update` 返回 `void`，没有“拒绝更新、要求上层重建”的返回通道。因此正式性能路径必须先落在 `graph_compute` 内部的 FlagOS Plan Builder/Cache，不能依赖外部 Graph Plan 生命周期。

内部主路径：

- `graph_compute`：查询内部 Plan Cache；Miss 时执行图分析、融合、Lower 和 Workspace 规划；Hit 时校验结构和 Binding 后执行；
- Shape、布局、静态参数或 Provider Capability 变化：内部创建新 Plan；
- 只有地址或动态 Scalar 变化：Provider 支持安全更新时更新 Binding，否则创建新 Plan/Native Graph；
- Cache 持有 Canonical Metadata 和 Provider Executable，不依赖短生命周期 `ggml_cgraph` 容器本身。

同时实现可选 GGML 回调，为显式使用该 API 的调用方提供加速：

- `graph_plan_create`：图分析、融合、Lower、Workspace 和 Provider Plan 创建；
- `graph_plan_update`：仅处理同拓扑且可安全更新的情况；检测到结构变化时在该 Plan 对象内部完成重建，因为接口不能返回失败；
- `graph_plan_compute`：执行已创建的 Plan；
- `graph_plan_free`：释放 Provider Executable、Native Graph 和 Workspace；

不能在每次 `graph_compute` 中重新做完整 Pattern Match。当前原生 CPU 在执行循环里检测 RMSNorm+MUL，并有 TODO 要求移到 Plan 阶段；FlagOS 应直接从 Plan 阶段起步。

### 11.8 Plan 所有权和生命周期

- Plan 必须拥有 Canonical Node/Edge、Pattern 结果、Typed Binding Template 和 Workspace 描述；
- Plan 不应浅拷贝 `ggml_cgraph` 后假设其 Node 数组永远有效；
- 可以在执行 Binding 中暂存本次调用的 `ggml_tensor *`，但不能在无法证明生命周期时把它作为长期所有权对象；
- Plan Cache 以 Canonical Structural Fingerprint 为主键，Hash 命中后必须做完整结构比较；
- Graph UID 不是跨 `sched_split_graph` 的稳定身份。它只能作为“调用方复用了同一个已分配图，可跳过部分结构校验”的 fast path，并必须正确处理 `uid == 0` 的 graph view；
- 裸指针只能参与本次 Binding Fingerprint，不能单独作为跨图结构身份；
- Native Graph、Module、Kernel、Workspace 和 Provider Context 的销毁顺序由 Plan 所有权明确约束。

### 11.9 `graph_optimize` 与内存收益边界

GGML Scheduler 会在图分割后、图分配前调用 Backend 的 `graph_optimize`。该接口收到的 split graph 是原图 node 数组的 view：原地置换节点顺序会影响后续 graph copy，因此适合做不会改变语义的重排；现有 Metal/CUDA/Hexagon Backend 已经使用这一入口。

但在当前 Scheduler 代码路径下，修改 view 的 `n_nodes` 或试图从 view 删除节点，不能减少 GGML Allocation。后续 graph copy 仍按原始图的 `[i_start, i_end)` 复制全部节点，Allocator 仍为原始中间 Tensor 计算生命周期并分配。该限制是当前实现事实，不作为 M3 内部技巧绕过。

首版 FlagOS Fusion Planner 仍以 `graph_compute` 内部 Plan 为主：

- 可以跳过被融合中间结果的写入；
- 可以减少实际内存流量、Kernel Launch 和 Barrier；
- 可以统一 Provider Workspace；
- 但不能自动收回 GGML 已经为原始中间 Tensor 预留的 Buffer。

若 Pattern 使用 `ggml_can_fuse()` 这类连续下标检查，候选节点必须在 node 数组中相邻；M3 必须把 `graph_optimize` 的合法重排作为此类 Pattern 的前置能力。对于本来已经相邻、或通过任意节点索引调用 `ggml_can_fuse_subgraph_ext()` 的 Pattern，则不应为了形式统一而强制重排。

如果目标是降低 GGML 分配峰值，只能在 llama.cpp `build_graph` 阶段不生成该中间节点，或修改/上游化 Scheduler 的 graph copy 与分配协议。该工作必须单独设计，因为它会影响 Scheduler Graph Copy、Allocator 生命周期、Graph Output、View/Alias 和 Debug 可观测性，不纳入 M3。

因此验收必须分开记录：

```text
intermediate bytes not read/written
provider workspace bytes
GGML allocated peak bytes
actual device resident peak bytes
```

M3 的 `graph_optimize` 只执行经过依赖和 Alias 校验的保守重排/标记；它是邻接型融合的正式交付能力，但不以减少 GGML Allocation 为交付承诺。

## 12. Native Graph 与公共 Plan 的关系

公共层只定义：

```text
prepare
bind/update
execute/replay
invalidate
destroy
```

Provider 映射：

| Provider | Execution Plan 实现 |
| --- | --- |
| Denglin | CUDA-compatible Graph Capture/Replay |
| NVIDIA | 原生 CUDA Graph Capture/Replay/Update |
| AMD | HIP Graph Capture/Replay；不支持的 Runtime/Library 组合回退到 Direct Plan |
| SpacemiT | AI Core 静态计划、Local Memory/TCM 和 DMA 调度 |
| ARM | 融合宏内核序列、线程计划、Prepack 和 Workspace Cache |

当前 Denglin Graph 使用 Tensor identity、地址、Shape、Stride、Op Params 和 resolved pointer 验证 Replay。重构时保留该安全性：

- Structural Fingerprint 判断 Plan 结构是否可复用；
- Binding Fingerprint 判断当前地址是否可直接 Replay；
- 没有 Graph Update 能力时，地址变化必须重新 Capture；
- 只有 Provider 实现参数更新后，才允许跨地址复用同一 Native Executable。

结构指纹只用于快速定位候选 Plan，不能单独证明可复用。Hash 命中后仍需比较 Canonical Graph；Binding Fingerprint 命中后仍需确认 Allocation Generation 和 Provider 约束。

设备支持 Native Graph 不等于所有节点都可捕获。每个 Execution Choice 必须声明 `capture_safe`，整个 Region 全部安全时才进入捕获。

## 13. ARM Reference Provider 设计

### 13.1 定位

ARM Provider 是：

- FlagOS Backend 在无目标加速卡环境中的可运行实现；
- AOT Package、Selector、Fusion Planner 和 Plan Cache 的参考平台；
- Demo、CI、开发和问题复现平台；
- 与 GPU/K3 融合语义进行差分测试的正确性基线之一。

它不是用来替代 llama.cpp 原生 CPU Backend，也不承担目标芯片最终性能承诺。

### 13.2 运行时

- Device Type：`ACCEL`。这是为了保留真正的 `ggml-cpu` 作为 Scheduler 末级 Fallback，不能改成 `CPU`；
- Memory：Host/Coherent，优先支持 Host Pointer；
- Queue：首版同步执行；
- Event：首版不提供或只提供真实的线程任务完成事件；
- Executable：静态函数、ELF `.so` 或直接链接的 AOT 函数表；
- 并行：Backend 初始化时创建持久线程池；
- ISA：运行时检测 NEON、DOTPROD、I8MM、SVE/SVE2、BF16、SME；
- Weight：按 Variant 做 Prepack，使用 Allocation Generation 管理失效；
- Workspace：按 Plan 一次分配，按线程切分 Scratch。

ARM Reference 必须显式启用，默认不在 GPU/K3 部署中枚举。原因是 llama.cpp 对所有非 CPU Backend 检查 `async + events`；一个同步 `ACCEL` Backend 会关闭整个模型的 pipeline parallel。首版采取以下策略：

- `GGML_FLAGOS_ARM` 构建选项默认关闭，开发/CI 镜像显式打开；
- 即使已编译，也只有 `FLAGOS_ENABLE_ARM_PROVIDER=1` 时才枚举 ARM Device；
- 与多 GPU 同时启用时打印 pipeline parallel 影响提示；
- 后续只有实现真实线程任务 Event 和异步 Queue，并通过并发测试后，才声明 `async/events=true`。

### 13.3 ARM 图融合执行

ARM 不执行 Graph Capture，而是执行已经 Lower 的计划：

```text
Plan Region 1: fused RMSNorm+Mul
  barrier
Plan Region 2: fused QKV projection
  barrier
Plan Region 3: RoPE+KV store
  barrier
Plan Region 4: online attention
```

每个融合 Region 内按 Row 或 Tile 分配线程，避免原始 GGML 每个节点一次全线程 Barrier。

Decode 和 Prefill 使用不同 Parallel Policy：

- Decode 单 Token：GEMV/Reduction 按输出通道或向量段切分；
- Prefill 多 Token：GEMM 和 Elementwise 优先按 Row/Tile 切分；
- 小 Tensor：限制线程数，避免同步成本超过计算；
- 大 Reduction：允许线程局部归约后做一次 Region 内同步。

### 13.4 ARM 融合优先级

1. `RMS_NORM + MUL`：先以多输出版本验证公共 Pattern、双阶段并行归约和 Plan；只有合法性检查证明 norm 无外部消费者时，才选择不写 norm 输出的中间结果消除 Variant；
2. `MATMUL/GEMV + ADD/BIAS/RESIDUAL`：在累加器仍位于向量寄存器时执行 Epilogue；
3. `SILU + MUL` 和 FFN Gate/Up：联合 Weight Prepack，复用输入激活；
4. QKV Projection：联合 Prepack 和一次调度产生多输出；
5. Q/K RoPE + KV Cache Write：减少变换和写回；
6. Online Attention：避免物化完整 Attention Score Tensor。

ARM 的主要验收不是与 GPU 比速度，而是：同一 Pattern/Plan 语义可以执行、关闭融合后结果一致、能够量化减少的内存流量和 Barrier。

## 14. SpacemiT K3 Provider 共建设计

### 14.1 定位

SpacemiT 是 `ggml-flagos` 下的正式 AI Chip Provider，不使用本项目的 `ggml-cpu/spacemit` 路径。现有 CPU 路径可以作为 Kernel、Repack、线程亲和性和 TCM 行为参考，但不是正式架构边界。

K3 向 GGML 返回 `GPU` 类型，使其进入 `model->devices` 并能正式承载模型层、权重和 KV Cache。这里的 `GPU` 只是 llama.cpp 当前调度协议中的“模型计算设备”类别；K3 的真实 AI Core、内存和队列能力由 `flagos_device_caps.kind` 及 Provider Contract 描述。首版不返回 `ACCEL`，否则只能作为 CPU Buffer Type 的机会性加速器；也不默认返回 `IGPU`，避免与独立 GPU 共存时被枚举策略忽略。

### 14.2 分工

FlagOS Core 负责：

- GGML 接入和 Backend 注册；
- Op Signature、Pattern、合法性检查；
- Fusion Planner、Plan 生命周期和 Cache；
- Fallback、诊断、测试和统一性能指标；
- Provider Contract 和版本兼容。

FlagOS 与进迭时空共建 Provider，进迭时空重点负责：

- K3 Device 发现、初始化和销毁；
- 内存域、Buffer、Copy、Local Memory/TCM 和 DMA；
- AI Core/线程/队列调度；
- AOT Module Loader 和 Typed Launcher；
- Kernel Variant、Shape/DType/Layout 约束；
- 厂商库和高性能宏内核；
- K3 Cost Model、Local Memory 预算和性能调优；
- Provider 级遥测。

### 14.3 K3 融合

公共 Pattern 进入 K3 Provider 后，由 Provider 决定：

- Lower 成一个 AOT Fused Kernel；
- Lower 成多个 Kernel 的静态 Command Plan；
- 使用 Local Memory/TCM 做 Tile Pipeline；
- 调用厂商图执行或高性能库；
- 因 Shape、内存或并行度原因拒绝融合，退回 Direct Op Plan。

首批共同 Pattern 建议与 ARM/Denglin 对齐：

- RMSNorm+Mul；
- MatMul+Add/Residual；
- FFN Gate/Up+SwiGLU；
- QKV；
- RoPE+KV Store；
- Attention Region。

Provider 必须输出拒绝原因，例如：

```text
unsupported_dtype
shape_constraint
local_memory_exceeded
external_consumer
layout_mismatch
insufficient_parallelism
missing_aot_variant
```

## 15. GPU Provider 接入流程与参考目标

1. 实现 Device/Memory/Queue/Event 基础能力；
2. 如果兼容 CUDA ABI，复用内部 `cuda_compat` 辅助层，但公共 Core 不包含 CUDA 类型；
3. 构建目标架构 AOT Package 和 Typed Binding；
4. 注册 Kernel Catalog 和 Selector；
5. 对接 BLAS、Attention、Quant 等厂商库；
6. 运行 Module Resolve + Scalar ABI + 真 Kernel Launch 自检；
7. 跑全部已声明支持的 `test-backend-ops`；
8. 跑模型端到端并记录未支持 Signature；
9. 接入 Fusion Pattern；
10. 最后接入 Native Graph，验证 Capture Safety、更新、失效和 Replay。

不能因为 Runtime API 名称与 CUDA 相同就默认行为一致，尤其要验证：

- Stream Capture 是否支持厂商 BLAS；
- Module Kernel 和异步 Copy 是否可捕获；
- Graph Instantiate/Replay 的地址稳定要求；
- Event、错误传播和多线程上下文行为。

### 15.1 NVIDIA CUDA Provider：RTX 3080 Reference Target

RTX 3080 的官方 CUDA Compute Capability 是 8.6，因此首个 Package 目标固定为 `sm_86`，不复用 Denglin cubin，也不经过 Denglin 的 CUDA-compatible ABI 层。Provider 内部直接使用 NVIDIA CUDA Driver/Runtime：

- `CUdevice/CUcontext/CUstream/CUevent` 或对应 Runtime Handle；
- `CUmodule` 加载 `sm_86` cubin，Package 可附 PTX 作为显式兼容 Fallback，但性能验收使用精确架构 cubin；
- Q4_K/Q6_K Direct AOT 先复刻已验证的 Denglin Signature，再接 cuBLAS/cuBLASLt、Attention 和量化库 Variant；
- Native Plan 使用 CUDA Stream Capture，必须在模型 warmup 后验证所有所选 Library Variant 的 capture safety；
- Device Query 决定实际显存预算，不能在方案里假定用户的 3080 是 10 GB 还是 12 GB 版本；Selector 根据实际 free/total memory 决定 Direct Quant、Repack Quant 或 Dequant Cache。

RTX 3080 是 Provider Contract 的离散 GPU 对照：它验证标准 CUDA 与 Denglin CUDA-compatible Runtime 能共享公共 Signature/Plan/Fusion，但保持各自 Module、Library、Graph 和错误语义。

### 15.2 AMD HIP Provider：Ryzen AI 9 HX 370 / Radeon 890M Reference Target

Radeon 890M 是 RDNA 3.5、16 CU、LLVM target `gfx1150`。当前 AMD ROCm 兼容矩阵已经列出 Ryzen AI 9 HX 370 / Radeon 890M；仍必须把具体 OS、Kernel、ROCm 版本作为测试矩阵的一部分，而不是只检查 `hipGetDeviceCount()`。

该 Provider 使用原生 HIP/ROCm，不用 CUDA ABI 包装：

- `hipDevice/hipStream/hipEvent`、HSACO/code object 与 `gfx1150` AOT Package；
- 基础 GEMM 接 rocBLAS/hipBLASLt，Q4_K/Q6_K 优先使用 FlagTree/TLE AOT Direct Quant Kernel；只有实际库版本声明并通过 INT4 group-quant 数值测试后才注册厂商量化 Variant；
- HIP 提供 Stream Capture、Graph Instantiate 和 Graph Launch API，但 Provider 必须逐 Variant 测 capture safety、地址更新与失败恢复，不能把 CUDA Graph 的测试结论直接复制过来；
- 890M 使用系统内存与动态/carveout 显存。Memory Provider 同时描述可用 Budget、Host Visibility、Coherency 和 Residency，不把 APU 简化成“无限统一显存”；Weight Cache/Workspace 必须受系统内存压力和 carveout 配置约束；
- 首阶段验收 F16/F32 与 Direct Q4_K，随后再以实际 HX 370 数据决定 hipBLASLt、融合宏内核和 HIP Graph 的 Selector 优先级。

Vulkan 只能作为以后独立的通用 GPU Provider 或部署 Fallback，不与 AMD HIP Provider 混成一个 Launcher；这样 ROCm 可用时保留完整的 AOT、Library 和 Graph 能力，不可用时也不会伪装成相同 Capability。

### 15.3 官方能力依据

- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)：RTX 3080 为 Compute Capability 8.6。
- [NVIDIA CUDA Graph Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)：定义显式 Graph 与 Stream Capture、Instantiate、Launch 生命周期。
- [AMD Ryzen AI 9 HX 370 产品规格](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html)：集成 Radeon 890M、16 Graphics Cores。
- [AMD ROCm GPU/APU 规格](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)：Radeon 890M 为 RDNA 3.5 / `gfx1150`，采用 Dynamic + carveout memory。
- [AMD ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)：用于固定 HX 370 的受支持 OS、Driver 与 ROCm 组合。
- [AMD HIP Graph 文档](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)：定义 Capture、Instantiate、Launch 与重复执行边界。

## 16. Scheduler、Fallback 和跨 Provider 执行

### 16.1 Fallback

```text
supports_op = false
    -> GGML Scheduler不把节点分给该FlagOS Device
    -> CPU或其他Backend执行
```

一旦 Scheduler 已把子图分给 FlagOS，Runtime Failure 不再进行隐式单节点 CPU Fallback。

只有融合实现、没有 Direct Op 实现的 Pattern 存在 Scheduler 风险：Scheduler 是按节点询问 `supports_op`，不能保证两个节点一定共同进入 Backend。因此首版要求融合 Pattern 中的每个计算节点都有独立执行路径；融合是优化，不是正确性的唯一实现。

### 16.2 跨 Provider

默认不在一个 FlagOS 子图内混合 Provider。每个 Scheduler Split 对应一个 Device/Provider。跨 Provider 的 Tensor 传递通过 GGML Backend Copy 接口完成，并遵守 Memory Domain 规则。

ARM Provider 不自动成为 SpacemiT/GPU 的内部 Fallback。需要回退时由 GGML Scheduler 选择 ARM FlagOS Device 或原生 CPU Backend，避免 Backend 内部隐藏数据迁移。

## 17. 构建、目录和部署

建议目录：

```text
ggml/src/ggml-flagos/
  ggml-flagos.cpp
  ggml-flagos.h

  core/
    flagos-provider.h
    flagos-device-registry.*
    flagos-op-signature.*
    flagos-kernel-catalog.*
    flagos-selector.*
    flagos-graph-analyzer.*
    flagos-fusion.*
    flagos-execution-plan.*
    flagos-memory-plan.*
    flagos-diagnostics.*

  providers/
    denglin/
      denglin-runtime.*
      denglin-memory.*
      denglin-launcher.*
      denglin-graph.*
      denglin-libraries.*

    nvidia/
      nvidia-runtime.*
      nvidia-memory.*
      nvidia-launcher.*
      nvidia-graph.*
      nvidia-libraries.*

    amd/
      amd-hip-runtime.*
      amd-memory.*
      amd-launcher.*
      amd-graph.*
      amd-libraries.*

    spacemit/
      spacemit-runtime.*
      spacemit-memory.*
      spacemit-launcher.*
      spacemit-plan.*
      spacemit-libraries.*

    arm/
      arm-runtime.*
      arm-features.*
      arm-threadpool.*
      arm-launcher.*
      arm-prepack.*
      arm-plan.*

  fusion/
    patterns-normalization.*
    patterns-matmul-epilogue.*
    patterns-qkv.*
    patterns-ffn.*
    patterns-attention.*

  kernels/
    denglin/<target>/
    nvidia/sm_86/
    amd/gfx1150/
    spacemit/<target>/
    arm/<target>/

  tests/
    test-provider-contract.*
    test-selector-consistency.*
    test-fusion-legality.*
    test-plan-cache.*
    test-aot-abi.*
```

CMake Provider 开关：

```text
GGML_FLAGOS=ON
GGML_FLAGOS_DENGLIN=ON/OFF
GGML_FLAGOS_NVIDIA=ON/OFF
GGML_FLAGOS_AMD=ON/OFF
GGML_FLAGOS_SPACEMIT=ON/OFF
GGML_FLAGOS_ARM=OFF/ON
```

没有 SDK 的 Provider 不参与构建，但 ARM Reference Provider 应尽可能在标准 AArch64 环境可构建。ARM 默认不构建；即使构建，也通过 `FLAGOS_ENABLE_ARM_PROVIDER=1` 显式枚举，避免无意影响 GPU pipeline parallel。

运行时支持 `FLAGOS_<PROVIDER>_MAX_DEVICES=N` 限制各 Provider 的枚举数量。Registry 必须在超过 `GGML_SCHED_MAX_BACKENDS` 前给出可读诊断，并保留原生 CPU 和非 FlagOS Backend 的容量。

## 18. 诊断和可观测性

统一诊断建议：

```text
FLAGOS_DUMP_PLAN=1
FLAGOS_FUSION=0/1
FLAGOS_FUSION_STATS=1
FLAGOS_LOG_UNSUPPORTED_OPS=1
FLAGOS_LOG_KERNELS=1
FLAGOS_SYNC_EACH_OP=1
FLAGOS_GRAPH_CAPTURE=0/1
```

Plan Dump 至少输出：

```text
Provider: FlagOS:ARM:0
GGML nodes: 312
Plan nodes: 147
Fused regions: 63
Estimated reads eliminated: ...
Estimated writes eliminated: ...
GGML allocated peak: ...
Actual device resident peak: ...
Barriers before/after: 286/121
Workspace: ...
Plan cache hit: ...
Native graph captures/replays: ...
Fusion reject reasons: ...
```

统计使用 `implementation_id` 作为索引，不再为每个 Kernel 在 Context 中增加一个手写计数器。

## 19. 测试与验收

### 19.1 Provider Contract

- Device 枚举和 Identity 唯一性；
- Device Type 调度语义：K3 能进入模型设备列表，ARM 不会被 `dev_by_type(CPU)` 选中，原生 CPU 始终保留为末级 Backend；
- 单个 Provider 探测失败、零设备和显式 required 失败三种行为；
- 总 Backend 上限和每 Provider 设备数限制的可读诊断；
- Buffer 分配、对齐、Copy、Host 可见性；
- Queue/Event 的真实语义；
- 错误码、资源销毁和多 Backend Context；
- 同 Provider、跨 Device和跨 Provider Copy 拒绝规则；特别覆盖 `Denglin:0 -> ARM:0` 不得进入 D2D Copy。

### 19.2 AOT ABI

- Manifest Schema；
- Runtime ABI 和 Kernel ABI Hash；
- Module 加载和全部符号解析；
- Scalar-heavy Kernel 真启动；
- 每个 Variant 的 dtype/shape/layout/alignment 约束边界、边界内和越界拒绝；
- 错误 Package、错误架构和错误版本必须在初始化时失败。

### 19.3 Selector 一致性

对所有 `supports_op=true` 的 Signature：

- 必须返回唯一合法 Execution Choice；
- 必须能 Bind；
- 必须存在 Workspace；
- 必须能执行或在 Provider 初始化阶段提前失败；
- 不允许执行路径出现额外 Shape 特判后拒绝。

### 19.4 Fusion

- Pattern 正匹配和反匹配；
- 多消费者、Graph Output、View/Alias、不同 Shape；
- 融合开关前后结果对比；
- 多输出融合；
- 动态 Shape 和地址更新；
- Workspace/Local Memory 超限退化；
- Pattern 重叠选择；
- Provider 拒绝后正确回到 Direct Op Plan。

### 19.5 Plan 和 Graph

- Plan Create/Update/Compute/Free；
- Structural Fingerprint 和 Binding Fingerprint；
- Native Graph Capture/Replay/Recapture；
- LRU 淘汰和资源释放；
- Capture Failure 后 Stream/Queue 状态恢复；
- 多 Shape、Batch、Context、并发请求。
- 相同结构不同 UID 命中同一 Plan 候选，以及 `uid == 0` graph view；
- `graph_optimize` 重排后的依赖正确性、邻接型 Pattern 命中，以及 GGML Allocation 口径保持独立；
- ARM 未启用时不影响多 GPU pipeline；ARM 同时启用但缺少 async/events 时产生明确诊断。

### 19.6 端到端

至少覆盖：

- Qwen2.5-1.5B 当前 Denglin 基线；
- 一个 Llama 系列模型；
- 一个包含 GQA/长上下文的模型；
- ARM Reference Provider 端到端 Demo；
- SpacemiT 选定模型；
- Prefill、Decode、长上下文和 Server；
- greedy 输出和数值容差；
- CPU Fallback 和全量 Offload。

### 19.7 性能指标

- PP/Prefill token/s；
- TG/Decode token/s；
- TTFT、TPOT；
- P50/P95/P99 和抖动；
- 峰值内存、Weight Cache、Workspace；
- 中间内存读写流量，以及与 GGML 分配峰值的区别；
- Kernel/Plan Node 数；
- Barrier/Launch 数；
- Capture、Replay、Recapture；
- 每个融合 Pattern 的命中率和净收益。

## 20. 分阶段实施计划

### M0：冻结 Denglin 基线

交付：

- 固化当前 392 个已声明支持测试；
- 固化 Qwen2.5 PP/TG/TTFT/长上下文数据；
- 固化 Kernel、Graph、Fusion 和 Cache 统计；
- 保存融合关闭和 Graph 关闭的对照结果。

退出条件：后续结构重构可以证明功能和性能没有无解释回退。

### M0.5：Decode 主路径数据门

现有 `tu-probe/fused_gemv.cu` 表明合并 4-bit 访存的 fused dequant-GEMV 有潜力，但它使用简化量化 Block，且 F16 对照是朴素 GEMV，不足以证明其已经击败当前 GGML Q4_K/Q6_K Direct AOT 或 Denglin BLAS 路径。因此本里程碑是“实现并决策”，不是预设替换结论。

交付：

- 用真实 GGML Q4_K/Q6_K Layout 实现 coalesced fused dequant-GEMV Variant；
- 在实际 Decode Shape 上比较四条路径：现有 Direct Quant AOT GEMV、Dequant Cache + Denglin BLAS、DLBLAS Repacked INT4 Group-Quant GEMV、新 Fused Dequant-GEMV；
- 记录 Kernel 时间、有效带宽、数值误差、临时/常驻显存和 Qwen TG32/长上下文 TPOT；
- Selector 按 Shape、Batch、Quant Type 和设备能力选择，不能用一个 Microbenchmark 全局替换；
- Fused 路径命中的 Weight 不再建立 F16 Dequant Cache，验证实际设备常驻显存下降。

退出条件：提交可复现实验和选择结论。新路径只有在端到端 TG 相对 M0 稳定提升至少 10%、P95 TPOT 不回退且数值通过时才成为对应 Signature 的默认选择；否则保留为实验 Variant，并根据 Profile 把 Attention、KV 或 Launch 瓶颈列为下一优化项。M0.5 与 M1 接口设计可并行，但默认路径决策必须在 M1 行为基线封板前完成。

当前检查点：DLBLAS Repacked Q4_K 已通过数值、Qwen3-4B 端到端和 CUDA Graph 验证，暖态 TG16 相对 Direct AOT 提升 127%；但首次进程内 JIT 将 no-warmup TG16 降到 1.21 token/s，且增加约 1.83 GiB 常驻权重。因此它满足“稳态性能 Variant”条件，尚不满足“默认部署 Variant”的冷启动/AOT 打包条件。Q6_K 与自有 Coalesced Fused Kernel 仍待完成。

负向测试还确认当前 `libdleol` 在 JIT 无法生成目标 CU function 时会直接终止进程，而不是可靠返回 DLBLAS status。当前代码因此在调用厂商库前要求显式的 `dlcc` 和 compile-options 预检，不满足则完全不注册该 Variant、直接使用 AOT；Provider 化后应把这种约束放入 Manifest/Capability Probe，并由独立 helper process 做隔离式真启动自检。

### M1：抽取 Provider 基础层，行为不变

交付：

- Provider Identity、Device、Memory、Queue、Module、Execution 接口；
- Device Type 策略落地并测试：正式模型设备使用 `GPU`，ARM Reference 使用 `ACCEL`，原生 `CPU` Backend 不被劫持；
- Denglin 代码迁入 `providers/denglin`；
- Core 中不再出现 CUDA 类型；
- `buft_is_flagos` 和 `buffer_copy_tensor` 改为校验 Provider ID、Device UUID、Memory Domain，覆盖相同 ordinal 的跨 Provider 拒绝；
- Provider 探测失败隔离、零设备合法、required Provider 失败语义；
- Provider Device 数限制和 `GGML_SCHED_MAX_BACKENDS` 前置检查；
- Denglin Device 暂时保留当前 `FlagOS0` 名称；
- `ggml-flagos` 仍只启用 Denglin，测试结果保持一致。

退出条件：Denglin 全部测试和端到端基线通过。

### M2：统一 Signature、Catalog 和 Selector

交付：

- Canonical Op Signature；
- Manifest v3 和生成的 Typed Binding；
- Manifest 的 dtype/shape/layout/alignment/capability 约束直接驱动 Selector，并自动生成每个 Variant 的边界拒绝测试；
- 固定成员 Kernel Registry 改为 Variant Catalog；
- `supports_op` 与执行共享 Selector；
- AOT Package 运行时搜索和 ABI 自检；
- Device 正式命名迁移为 `FlagOS:<Provider>:<ordinal>`，同批更新 `validate_denglin.sh` 等仓库脚本、CI、配置模板和文档；若未扩展名称解析协议，不声明不存在的旧名 alias。

退出条件：不存在 `supports_op=true` 但执行选择为空的测试用例。

### M3：Graph Plan 和公共 Fusion Planner

交付：

- `graph_compute` 内部 Plan Builder/Cache，并补齐可选的 `graph_plan_create/update/compute/free`；
- Graph Analyzer、Pattern Registry、合法性检查、Plan Cache；
- Structural Fingerprint 作为 Cache 主键，UID 仅作同一已分配图的 fast path；
- `graph_optimize` 交付依赖安全的节点重排，为邻接型 Pattern 制造邻接性，但不承诺减少 GGML Allocation；
- 当前 RMSNorm+Mul 多输出 Launch 合并迁入公共 Pattern，用于验证 Registry/Plan；新增只写 mul 的 Variant 时同批接入 `ggml_can_fuse_subgraph_ext()` 合法性检查；
- Denglin CUDA Graph 成为 Provider Native Plan；
- Plan Dump、融合统计和 Reject Reason。

退出条件：融合开关数值一致，Denglin Graph Capture/Replay 保持正确；无论调用方是否使用 GGML `graph_plan_*`，稳定图的 Plan 构建都不再位于 Token 热路径。

当前实现检查点（2026-08-22）：已先在现有 Denglin Backend 中落下 M3 的最小纵向切片，包括无 CUDA 类型的 Canonical Graph、Structural Fingerprint + 有界 LRU Plan Cache、Pattern Candidate、Provider `can_lower/cost/capture_safe`、非重叠 Greedy 选择，以及 RMSNorm+Mul 从执行循环特判迁移到 Plan。Denglin 仍复用原 AOT 多输出 Kernel；未实现的 Catalog Pattern 一律拒绝。该检查点通过独立 Planner 单测和 Denglin 已声明支持的 392 项算子测试，但不代表 M1-M3 已整体完成：Provider 基础层迁移、统一 Signature/Selector、依赖安全重排、完整合法性/Reject Reason、GGML `graph_plan_*` 回调仍按本计划继续推进。

### M4：ARM Reference Provider

交付：

- Host Buffer、`ACCEL` Device、同步执行和持久线程池；
- 构建和运行时双重显式启用，默认不影响 GPU/K3 部署的 pipeline parallel；
- AArch64 ISA 探测和 AOT ELF Package；
- Direct Op 基础覆盖；
- RMSNorm+Mul、MatMul Epilogue、SwiGLU 首批融合；
- Qwen/Llama 小模型端到端 Demo。

退出条件：无 GPU 环境可运行同一 `ggml-flagos`，融合开关正确，Plan 和 AOT ABI 可以作为 CI 基线。

### M5：SpacemiT K3 Provider 共建

交付：

- `GPU` 调度类型的 Device，以及 Memory/Execution/Local Memory Provider；
- AOT Package 和 Kernel Catalog；
- Direct Op 端到端；
- K3 Static Plan 和首批融合；
- 目标模型性能报告。

退出条件：选定模型端到端正确，Fallback 边界明确，核心 Pattern 有逐项收益数据。

### M6a：NVIDIA CUDA Provider（RTX 3080）

交付：

- 原生 CUDA Runtime/Driver、`sm_86` AOT Package、cuBLAS/cuBLASLt；
- CUDA Graph Capture/Replay/Update 能力探测；
- Direct Op、融合和端到端；
- 与 Denglin 同一套测试矩阵。

退出条件：RTX 3080 上 Qwen3-4B Q4_K_M 端到端正确；新增 NVIDIA GPU 没有修改 GGML Core 接口和公共算子语义，仅增加 Provider、Package 和必要的公共能力扩展。

### M6b：AMD HIP Provider（HX 370 / Radeon 890M）

交付：

- 原生 HIP Runtime、`gfx1150` AOT Package、rocBLAS/hipBLASLt；
- APU Memory Budget/Residency 能力模型；
- HIP Graph Capture/Replay 能力探测与 Direct Plan Fallback；
- Direct Q4_K、基础融合和 Qwen 小模型端到端；
- 固定 OS、Kernel、ROCm、Driver 的可复现实验环境。

退出条件：Radeon 890M 上端到端正确且无隐藏 CPU 执行；共享公共 Signature/Plan/Fusion，没有把 HIP 类型泄露到 Core。

### M7：Common IR 评估

只有当以下条件满足后进入：

- 热点 Pattern 已通过手写/AOT 实现证明有收益；
- 多 Provider 出现相同的较大子图需求；
- 内存生命周期和 Layout 需求已经稳定；
- Common IR 能替代而不是重复现有 Plan/Fusion 能力。

## 21. 团队协作和接口治理

### FlagOS Core 变更

- FlagOS 主导；
- Provider 团队共同评审；
- Provider Contract 变更必须有版本和兼容说明；
- 新的公共 Pattern 必须包含语义、合法性、测试和至少一个 Provider 实现。

### SpacemiT Provider 变更

- FlagOS 与进迭时空共同维护；
- 进迭时空对 Runtime、Kernel、Local Memory、Cost Model 和性能负责；
- FlagOS 对 GGML 集成、Plan 生命周期、Fallback 和跨 Provider 一致性负责。

### AOT Package 发布

- Package、Manifest、编译器版本、ABI Hash 同批次发布；
- 每个 Package 附带目标架构和最小 Runtime 版本；
- 合入前运行统一 Conformance Suite。

## 22. 主要风险和控制措施

| 风险 | 后果 | 控制措施 |
| --- | --- | --- |
| 把 GGML Device Type 当物理类型 | K3 不承载模型层，ARM 劫持原生 CPU | K3=`GPU`、ARM=`ACCEL`，物理类型放 Provider Caps，调度回归测试 |
| 公共接口仍然 CUDA 化 | ARM/K3 需要大量伪实现 | Core 禁止 CUDA 类型，功能使用可选扩展 |
| 抽象过度 | 重构周期长，当前性能回退 | M1 只迁移 Denglin，行为不变；外部插件延后 |
| `supports_op` 与执行不一致 | Runtime Failure | 单一 Selector 和一致性测试 |
| 融合破坏图语义 | 静默错误 | 使用 GGML 合法性检查，多输出和 Alias 测试 |
| Native Graph 使用陈旧地址 | 错误结果或崩溃 | Allocation Generation、Binding Fingerprint、严格失效 |
| UID 被误作稳定 Plan Key | 重建图持续 Cache Miss 或错误复用 | Structural Fingerprint 主键、完整比较，UID 仅 fast path |
| AOT ABI 漂移 | 参数错位、非法访问 | Manifest v3、ABI Hash、Typed Binding、真启动自检 |
| 跨 Provider Device 0 混淆 | 错误 D2D Copy | Provider ID、Device UUID、Memory Domain |
| 同步 ARM ACCEL 关闭多 GPU 流水 | 整机吞吐回退 | ARM 默认不构建/不枚举；并存诊断；真实 async/event 后再开放 |
| Provider/Device 数超过 Scheduler 上限 | 裸断言退出 | Registry 前置容量检查、每 Provider 枚举限制、可读错误 |
| 过度融合 | 性能下降 | Cost Model、运行时开关、逐 Pattern 收益统计 |
| 把简化 GEMV 探针外推到真实模型 | 错误替换 Decode 主路径 | M0.5 用真实 Quant Layout、四路径和端到端 TG 做数据门 |
| 厂商量化库首次 JIT | TTFT/首轮 TPOT 严重回退 | Package AOT、持久 JIT Cache 或模型加载预热；冷态与暖态分别验收 |
| 把 CUDA Graph 结论复制到 HIP | Capture 失败或错误 Replay | AMD Provider 独立 capability probe、Library capture-safe 白名单和 Graph Fallback |
| 把 890M 共享内存当无限显存 | 系统抖动、OOM、吞吐不稳定 | 动态 Budget、Residency/Carveout 遥测、受限 Weight Cache 与压力测试 |
| ARM Demo 被误当性能承诺 | 项目目标失焦 | 明确 Reference 定位和独立验收口径 |
| SpacemiT 与 Core 同时快速演化 | 集成频繁破坏 | Contract 版本、共建评审、Conformance CI |

## 23. 近期建议执行项

近期不应继续向当前 3100 余行单文件直接增加新芯片分支。建议立刻执行：

1. 冻结并自动化当前 Denglin 基线；
2. 完成真实 Q4_K/Q6_K Decode 四路径对比，按 M0.5 数据门确定默认 Selector；
3. 设计并确认 Provider Contract、Device Identity、Device Type 策略和 Memory Domain；
4. 将 Denglin Runtime/Launcher/Graph 抽出，但不改变封板后的功能和性能；
5. 同批修复跨 Provider Copy 身份判断、Probe 隔离和 Scheduler 容量诊断；
6. 统一 Op Signature、Manifest 约束和 Selector；
7. 在 `graph_compute` 内实现公共 Plan/Fusion，并以 `graph_optimize` 支持邻接型 Pattern；
8. 用现有 RMSNorm+Mul 多输出版本作为第一个迁移 Pattern，先验证 Plan 通路，再评估中间结果消除 Variant；
9. 以显式启用的 ARM `ACCEL` Provider 验证同一 Pattern 在 CPU AOT 上的 Lower；
10. 与进迭时空并行定义 K3 `GPU` 调度设备、Memory、Local Memory 和 AOT Contract；
11. 用 RTX 3080 落地 NVIDIA `sm_86` Provider，验证标准 CUDA 对照；
12. 用 HX 370 / Radeon 890M 落地 AMD `gfx1150` HIP Provider，验证 APU Memory Domain 与 HIP Graph；
13. 再推进 K3、NVIDIA 和 AMD 的正式性能 Selector。

## 24. 最终判断

现有 Denglin 成果已经证明 `GGML -> ggml-flagos -> AOT/厂商库 -> 目标芯片` 路径可行，也证明了 CUDA Graph 能在稳定 Decode 子图上工作。下一阶段不应继续以“增加更多逐算子分支”为主，而应完成两项平台化工作：

1. 把 Denglin 专用实现收敛为多 Provider 的 FlagOS Core；
2. 把融合从执行循环特判升级为 Plan 阶段的公共图优化能力。

最终竞争力来自：

```text
稳定的GGML生态入口
+ 多芯片Provider接入效率
+ AOT Kernel与厂商库
+ Provider-aware图融合
+ Native/Static Execution Plan
+ 可验证的端到端性能
```

ARM Reference Provider 用来降低开发和验证门槛；Denglin、NVIDIA、AMD 和 SpacemiT K3 用来交付真实性能。SpacemiT 不是外部并列 Backend，而是 `ggml-flagos` 内与 FlagOS 共建的正式 AI Chip Provider。
