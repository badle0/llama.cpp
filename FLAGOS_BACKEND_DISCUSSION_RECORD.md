# FlagOS llama.cpp Backend 讨论记录

日期：2026-08-21
性质：对本轮讨论和已确认结论的结构化记录，不是逐字聊天导出
关联方案：`FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md`

## 1. 项目目标

FlagOS 不改变 llama.cpp 的 GGUF、KV Cache、模型架构建图和上层生态，而是通过标准 GGML Backend 接管目标芯片支持的计算子图：

```text
GGUF / llama.cpp模型建图
          ↓
GGML Scheduler
          ↓
ggml-flagos Core
          ↓
目标芯片Provider
          ↓
AOT Kernel / 厂商库 / Execution Plan
```

首版边界：

- 一个推理实例只选择一个 FlagOS 加速 Provider；
- 不考虑 Denglin、K3、新 GPU 之间同时执行同一模型；
- 不建设跨异构芯片切图、跨设备融合或跨 Provider 传输优化；
- 跨 Provider Copy 只保留身份校验和安全拒绝能力；
- 原生 `ggml-cpu` 始终保留为不支持节点的末级 Fallback。

## 2. 多平台 Provider 架构

只有一个 GGML Backend 家族：`ggml-flagos`。

```text
ggml-flagos Core
  ├── Device Registry
  ├── Canonical Op Signature
  ├── Kernel Catalog / Selector
  ├── Graph Analyzer / Pattern Registry
  ├── Fusion Planner / Cost Model
  ├── Execution Plan / Plan Cache
  └── Diagnostics / Tests
           ↓
      Provider Contract
           ↓
  ├── Denglin GPU Provider
  ├── New GPU Provider
  ├── SpacemiT K3 Provider
  └── ARM Reference Provider
```

已确认的 GGML Device Type：

- Denglin、新 GPU：`GPU`；
- SpacemiT K3：`GPU`，确保进入 `model->devices` 并承载模型层和 KV Cache；
- ARM Reference：`ACCEL`，避免劫持真正的 CPU Backend；
- 真实物理类别由 `flagos_device_caps.kind` 表示，不能把 GGML Device Type 当硬件分类。

SpacemiT 不建设并列的 `ggml-spacemit`，而是在 `ggml-flagos/providers/spacemit` 下由 FlagOS 与进迭时空共建。

ARM Provider 用于开发、Demo、AOT ABI、融合正确性和 CI，不承担目标芯片性能承诺。ARM 默认不构建、不枚举；同步 ACCEL Backend 会影响多 GPU pipeline parallel，因此必须显式启用。

## 3. GGUF、模型图和“相邻算子”

GGUF 主要保存权重、模型架构元数据、Tokenizer 和量化信息，通常不保存一张固定运行时算子图。

运行时图由 llama.cpp 根据以下信息构建：

- 模型架构；
- Prefill 或 Decode；
- Batch、Context Length；
- KV Cache；
- Flash Attention 配置；
- MoE Expert 选择等运行参数。

因此同一个 GGUF 可以产生不同 GGML 图。

“相邻算子”必须区分：

1. 数组相邻：`cgraph->nodes[i]` 和 `nodes[i+1]`；
2. 数据流相邻：后一个节点直接消费前一个节点输出；
3. 可融合子图：多个节点构成依赖闭合的 Region，中间结果没有 Region 外消费者。

融合应依据数据依赖和子图闭合性，不能假设模型文件已经把可融合节点物理相邻排列。`graph_optimize` 可以做依赖安全的节点重排，为需要连续下标的 Pattern 制造邻接性。

## 4. 算子融合、图融合和 CUDA Graph

### 4.1 算子融合

算子融合把多个 GGML 节点 Lower 成一个真实 Kernel：

```text
RMS_NORM + MUL
       ↓
一个Fused AOT Kernel
```

主要收益：

- 减少 Kernel Launch；
- 减少中间 Tensor 读写；
- 在寄存器、Shared Memory、Cache 或 TCM 中复用中间值；
- 把 Bias、Activation、Residual 等放入 GEMM/GEMV Epilogue。

算子融合的具体实现高度依赖芯片，由 Provider、FlagTree/TLE AOT Kernel 或厂商库完成。

### 4.2 图融合

图融合从完整 GGML 子图中识别一个可以整体优化的 Region：

```text
GGML Subgraph
    ↓
Canonical Graph
    ↓
Pattern匹配
    ↓
依赖、Use Count、Output、View、Alias检查
    ↓
Provider can_lower + Cost
    ↓
Execution Plan
```

图融合的结果不一定是一个 Kernel，也可能是：

- 一个融合 AOT Kernel；
- 厂商库调用加 Epilogue；
- 多个 Kernel 的静态 Command Plan；
- 共享 Workspace 和调度的多 Kernel Region；
- Provider 拒绝融合后回到 Direct Op Sequence。

因此：

```text
图融合负责识别和规划Region
算子融合是Region Lowering成单Kernel的一种结果
```

### 4.3 CUDA Graph

CUDA Graph 不做计算融合，只捕获已经确定的 Kernel Launch 序列并重复 Replay：

```text
图分析与融合
    ↓
Provider Lowering
    ↓
最终Kernel序列
    ↓
CUDA Graph Capture
    ↓
Decode Replay
```

四种组合的差别：

```text
无融合、无CUDA Graph：10个Kernel，CPU提交10次
只有图融合：          4个Kernel，CPU提交4次
只有CUDA Graph：      10个Kernel，一次Graph Replay
两者都有：            4个Kernel，一次Graph Replay
```

图融合改变“执行什么”；CUDA Graph 优化“如何重复提交”。正确顺序是先融合和 Lower，再 Capture。

没有 CUDA Graph 的芯片仍能做图融合：

- K3：AI Core Static Command Plan、TCM/DMA 调度；
- ARM：宏内核、持久线程池和缓存后的线程计划；
- 其他 GPU：Native Graph 或 Command Buffer。

## 5. llama.cpp 中的代码层次

### 5.1 llama 模型建图层

位置：

```text
src/llama-model.cpp
src/llama-graph.cpp
```

职责是生成语义正确的 GGML DAG。这里不应添加 Denglin、K3 等芯片专用融合分支。

只有希望真正不创建某个中间 Tensor、降低 GGML Allocator 分配峰值时，才可能需要修改建图层或上游 Scheduler 协议；这不属于首版 M3。

### 5.2 GGML Scheduler

位置：

```text
ggml/src/ggml-backend.cpp
```

职责：

- 根据 `supports_op` 和 Buffer Type 划分 Backend Split；
- 调用 Backend `graph_optimize`；
- 分配 Tensor 和处理必要 Copy；
- 调用每个 Backend 的 `graph_compute`。

Scheduler 不放 FlagOS 或芯片专用 Pattern。

### 5.3 ggml-flagos Core Plan 层

建议位置：

```text
ggml/src/ggml-flagos/core/
  flagos-graph-analyzer.*
  flagos-pattern-registry.*
  flagos-fusion-planner.*
  flagos-execution-plan.*
  flagos-plan-cache.*
```

主路径放在 `graph_compute` 内：

```text
Plan Cache Lookup
    ↓ Miss
Graph Analyzer
    ↓
Pattern Registry
    ↓
Fusion Legality
    ↓
Provider can_lower / Cost
    ↓
Region Selection
    ↓
Provider Lowering
    ↓
Bind + Execute
```

当前 `graph_plan_create/update/compute/free` 在 GGML 中标记为尚未使用，不能作为 llama.cpp 主路径依赖。可以补齐这些接口，但正式性能路径先放在 `graph_compute` 内部 Plan Cache。

### 5.4 Provider Lowering 和 Launcher

建议位置：

```text
providers/<chip>/
  <chip>-lowering.*
  <chip>-launcher.*
  <chip>-graph-or-plan.*
```

Provider 负责：

- `can_lower(region)`；
- Shape/DType/Layout/资源限制；
- 成本估计；
- Fused AOT Kernel 或厂商库；
- Typed Binding；
- Workspace、Queue、Event；
- Native Graph 或 Static Plan。

## 6. FlagOS 与厂商的融合分工

FlagOS Core 统一负责：

- Pattern ID 和语义；
- 图匹配；
- Use Count、Output、View、Alias 合法性；
- Region 重叠选择；
- Plan Cache；
- 统一测试、诊断和收益统计。

厂商 Provider 负责：

- 支持哪些 Pattern；
- 支持的 Shape、DType、Layout；
- 如何 Lower；
- 提供融合 AOT Kernel、厂商库或静态计划；
- 芯片侧性能优化。

每个厂商的融合集合可以不同：

```text
公共Pattern：MATMUL + ADD + SILU

Denglin：一个Fused AOT Kernel
K3：AI Core宏内核或Static Plan
ARM：MatMul Microkernel内执行Epilogue
新GPU：只融合MATMUL+ADD
某芯片：拒绝融合，三个Direct Op
```

每个融合 Pattern 中的计算节点仍必须存在 Direct Op 路径。融合是性能优化，不能成为正确性的唯一实现。

## 7. 当前实现事实

当前 Denglin Backend 已具备：

- GGML Backend、Device、Buffer、Stream、Event；
- Triton/FlagTree AOT Kernel；
- C++ Driver Launcher；
- Q4_K/Q6_K Direct Kernel；
- Dequant Cache + Denglin BLAS 路径；
- RMSNorm+Mul 单 Launch、多输出融合；
- CUDA-compatible Graph Capture/Replay；
- 端到端模型推理。

当前 RMSNorm+Mul 仍同时写 `norm_output` 和 `mul_output`：

- 减少一次 Launch；
- 减少 MUL 对 norm 结果的一次读取；
- 不消除 norm 写入；
- 不降低 GGML 对 norm Tensor 的分配。

当前融合逻辑仍是 `graph_compute` 执行循环中的 `nodes[i+1]` 特判。下一步应迁入公共 Pattern Registry、合法性检查和 Plan Cache。

当前 CUDA Graph 捕获的是 `flagos_graph_evaluate()` 已经决定的 Kernel 序列；CUDA Graph 没有产生 RMSNorm+Mul 融合。

## 8. 离散访存

离散访存是线程访问的地址不连续、跨度大或缺乏规律，也称非合并访存、Scatter/Gather 或 Uncoalesced Access。

GPU Warp 连续访问：

```text
线程0 → address 0
线程1 → address 4
线程2 → address 8
...
```

硬件可以合并成少量 Memory Transaction。

离散访问：

```text
线程0 → address 0
线程1 → address 128
线程2 → address 256
...
```

可能产生接近每线程一次独立事务，浪费 Cache Line 和带宽。

在量化 GEMV 中，应尽量让 Warp 连续读取 Quant Block，使用对齐的 `uint32`/Vectorized Load，在寄存器中拆解多个 4-bit/6-bit 值并复用 Scale。融合并不会自动解决离散访存；融合 Kernel 如果 Weight Layout 和线程映射不好，仍可能比逐算子更慢。

## 9. Decode 性能数据门

当前 Decode/TG 相比 CPU 基本没有形成明显优势，不能把“CUDA Graph 已 Replay”视为 Decode 已优化完成。

M0.5 应使用真实 GGML Q4_K/Q6_K Layout，对比：

1. 当前 Direct Quant AOT GEMV；
2. Dequant Cache + Denglin BLAS；
3. Coalesced Fused Dequant-GEMV。

记录：

- Kernel 时间和有效带宽；
- 数值误差；
- 临时和常驻显存；
- TG32、长上下文 TPOT 和 P95；
- Attention、KV、Launch 和 GEMV 的耗时占比。

只有端到端 TG 稳定提升、P95 不回退、数值通过且显存收益成立时，新路径才成为对应 Signature 的默认 Selector Choice。

## 10. 实施顺序

```text
M0    固化Denglin正确性和性能基线
M0.5 真实Q4_K/Q6_K Decode三路径数据门
M1    Provider抽取、Copy身份、Probe隔离和容量检查
M2    Signature/Catalog/Selector、Manifest约束和Typed Binding
M3    graph_compute内部Plan Cache、Graph Analyzer、Fusion Planner
M4    ARM ACCEL Reference Provider
M5    SpacemiT K3 GPU调度型AI Provider
M6    新GPU Provider
M7    Common IR评估
```

## 11. 核心结论

```text
GGML固定算子语义和数据依赖
FlagOS Core识别合法融合Region
芯片Provider决定是否融合以及如何Lower
Execution Plan组织最终Kernel序列
CUDA Graph只负责捕获和Replay该序列
```

图融合框架和合法性不应让每个厂商重复建设；不同厂商只需要实现自己能够高效执行的 Pattern Lowering。不同芯片支持的融合集合、Shape 范围和执行方式可以不同。
