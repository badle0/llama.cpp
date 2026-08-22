# Claude Code 设计评审处置记录

日期：2026-08-21
输入：`FLAGOS_BACKEND_MULTIPLATFORM_DESIGN_REVIEW.md`
修订目标：`FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` v0.2

## 1. 结论

Claude 的代码事实评审总体成立，已经用于修订主方案。五项架构问题直接采纳，两项建议按当前接口和实验边界做了修正后采纳：

| Review 项 | 处置 | 主方案结果 |
| --- | --- | --- |
| Device Type 是调度策略 | 采纳并收紧 | K3=`GPU`，ARM=`ACCEL`，物理类别留在 Provider Caps |
| Graph UID 不稳定 | 完全采纳 | Structural Fingerprint 为 Plan Cache 主键；UID 仅 fast path，支持 uid=0 |
| `graph_optimize` 不能删分配节点 | 完全采纳 | M3 只用它做合法重排/邻接，不承诺降低 GGML Allocation |
| RMSNorm+Mul 成果描述不准 | 采纳并校正计量 | 定义为单 Launch、多输出；省一次 read，不省 norm write/Allocation |
| 跨 Provider Copy 身份不足 | 完全采纳 | M1 强制 Provider ID + Device UUID + Memory Domain，并加 Denglin:0/ARM:0 反例 |
| 缺少 Decode 里程碑 | 修正后采纳 | 新增 M0.5 三路径数据门，不依据简化探针直接替换默认路径 |
| Device 旧名 alias | 不按原建议直接实现 | M1 保留 `FlagOS0`；M2 显式改名。当前没有 alias API，不声明虚假兼容 |
| Manifest/ARM Event/Backend 上限/Probe | 完全采纳 | 分别进入 M2、M4、M1 和 Provider Contract/Test |

## 2. 两个关键修正

### 2.1 K3 固定用 `GPU`，不在 `GPU/IGPU` 间含糊选择

K3 是正式模型执行设备，需要进入 `model->devices`、承载层和 KV Cache。`ACCEL` 只会进入 CPU Buffer Type 的机会性接管路径，不符合目标；`IGPU` 在独立 GPU 存在时可能被 llama.cpp 忽略。因此首版对 GGML 返回 `GPU`，真实 AI Core 类型由 `flagos_device_caps.kind` 描述。

ARM Reference 返回 `ACCEL`，避免注册顺序导致它被 `ggml_backend_dev_by_type(CPU)` 误选，确保原生 `ggml-cpu` 始终作为 Scheduler 末级 Fallback。同步 ARM ACCEL 会影响全模型 pipeline parallel，因此构建和枚举均默认关闭，仅在开发、Demo 和 CI 中显式启用。

### 2.2 M0.5 是数据门，不是预设替换

`tu-probe/fused_gemv.cu` 证明合并 4-bit 访存值得继续实现，但它使用简化量化布局，且其 F16 对照不是当前 backend 的真实 dlBLAS 路径，不能直接推出“当前 Decode 默认路径已被击败”。

M0.5 改为在实际 GGML Q4_K/Q6_K 和模型 Shape 上比较：

1. 当前 Direct Quant AOT GEMV；
2. Dequant Cache + Denglin BLAS；
3. Coalesced Fused Dequant-GEMV。

只有端到端 TG、P95 TPOT、数值和显存同时通过门槛，第三条路径才成为对应 Signature 的默认选择。否则保留为按 Shape 选择的实验 Variant，并继续按 Profile 判断瓶颈是否在 Attention、KV 或 Launch。

## 3. 融合口径

当前 RMSNorm+Mul Kernel 同时写 norm 和 mul 两个输出。因此：

- 消除一个 Kernel Launch；
- MUL 直接复用归一化值，消除一次 norm 中间结果读取；
- norm 输出写入仍存在；
- GGML 为 norm 中间 Tensor 的分配仍存在。

所以成本模型改为分别统计 `eliminated_read_bytes` 和 `eliminated_write_bytes`。未来只写 mul 的 Variant 必须与 `ggml_can_fuse_subgraph_ext()` 的 use count、Graph Output、View/Alias 检查同批落地。

当前 Scheduler 会从原始 split 区间重新复制节点，修改 graph view 的 `n_nodes` 无法让 Allocator 少分配中间 Tensor。降低 GGML 分配峰值需要修改建图或 Scheduler 协议，不属于 M3。M3 的 `graph_optimize` 价值是依赖安全的重排，为需要连续节点的 Pattern 制造邻接性。

## 4. 实施顺序变化

```text
M0    固化当前 Denglin 正确性和性能基线
M0.5 真实 Q4_K/Q6_K Decode 三路径实现、对比和 Selector 决策
M1    Provider 抽取 + Copy 身份修复 + Probe 隔离 + 容量检查
M2    Signature/Catalog/Selector + 可执行 Manifest 约束 + 命名迁移
M3    Structural Plan Cache + graph_optimize 重排 + 公共 Fusion Planner
M4    显式启用的 ARM ACCEL Reference Provider
M5    K3 GPU 调度型 AI Provider
M6a   NVIDIA CUDA Provider（RTX 3080 / sm_86）
M6b   AMD HIP Provider（HX 370 Radeon 890M / gfx1150）
M7    Common IR 评估
```

## 4.1 2026-08-22 后续实测增补

主方案 v0.3 已补入真实 Q4_K 到 DLBLAS INT4 group-quant 的重排路径。Qwen3-4B 暖态 TG16 从 Direct AOT + Graph 的 4.74 token/s 提升到 DLBLAS + Graph 的 10.75 token/s，但 no-warmup 首轮只有 1.21 token/s，额外权重缓存约 1.83 GiB。因此该 Variant 继续 opt-in，待厂商内核 AOT 化或持久 JIT Cache 解决后再进入默认 Selector。

“新 GPU”也已拆成 NVIDIA 与 AMD 两个正式参考 Provider。二者共享 FlagOS Core 的 Signature、Fusion Planner 和 Execution Plan Contract，但分别拥有 CUDA/HIP Runtime、Module、Library 和 Native Graph 实现，禁止进入 Denglin Launcher 增加条件分支。

## 5. 本次变更边界

本次只修订架构和实施方案，没有修改 `ggml-flagos.cpp`、Scheduler 或 Kernel。上述 M0.5/M1 之后的条目仍是待实现工作，不能把设计修订表述为代码已经完成。
