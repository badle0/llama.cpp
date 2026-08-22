# `FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md` 代码事实评审

评审日期：2026-08-21
评审对象：`FLAGOS_BACKEND_MULTIPLATFORM_DESIGN.md`（设计评审稿，1145 行）
评审方式：逐条对照当前工作树的 `ggml/src/ggml-backend.cpp`、`ggml-backend-impl.h`、
`ggml-backend-reg.cpp`、`ggml-alloc.c`、`ggml-impl.h`、`ggml.c`、`src/llama.cpp`、
`src/llama-context.cpp`、`src/llama-model.cpp`、`common/arg.cpp` 和
`ggml/src/ggml-flagos/ggml-flagos.cpp`

## 0. 总体判断

架构方向站得住：单一 `ggml-flagos` Backend 加多 Provider、`supports_op` 与执行共用
Selector、把融合从执行循环特判上移到 Plan 阶段，这三条是当前 2575 行单文件的正确出路。
方案自审中修正的两处（`graph_plan_*` 不是 Scheduler 主路径；执行期融合不自动降低 GGML
分配峰值）经代码核对**均为事实**，方向正确。

但有三处结论与当前 llama.cpp 代码冲突，其中第 1 处会导致 ARM Provider 一上线就破坏
CPU Fallback，必须在 M1 之前修正设计。另有一处对现有成果的描述名不副实，以及实施
计划缺少 decode 性能里程碑。

本文只记录与代码事实冲突或需要补强的部分，未列出的章节视为通过。

---

## 1. 必须修正：Device Type 是策略选择器，不是硬件描述

对应方案 §5（line 155）：

> 每个 Device 必须返回真实类型：Denglin/新 GPU 返回 `GPU`，SpacemiT K3 返回 `ACCEL`，
> ARM Reference 返回 `CPU`。

llama.cpp 用 `ggml_backend_dev_type` 做**分派决策**，不做展示。按"真实硬件形态"填写会
直接改变模型加载和 Fallback 行为。

### 1.1 K3 返回 `ACCEL`：该芯片不会承载任何层

`src/llama.cpp:220-221` 在枚举设备构建 `model->devices` 时，把 `CPU` 和 `ACCEL`
**一起 skip**：

```cpp
case GGML_BACKEND_DEVICE_TYPE_CPU:
case GGML_BACKEND_DEVICE_TYPE_ACCEL:
    // skip CPU backends since they are handled separately
    break;
```

不在 `model->devices` 中，`get_layer_buft_list()`（`llama-model.cpp:1343-1353`）就永远
不会把任何层指向它，`dev_layer(il)` 也不会返回它，因此 KV Cache
（`llama-kv-cache.cpp:214-217`）同样不会落在它的 buffer 上。

`ACCEL` 仍有一条权重路径：`make_cpu_buft_list()`（`llama-model.cpp:907-917`）会把 ACCEL
的 buft 放在 CPU buft list 的**最前面**，`select_buft` 因此可能选中它。但这是 BLAS
Backend 的机会性接管模式（`ggml-blas.cpp:355` 正是 `ACCEL`），不是"正式 AI 芯片
Provider 承载模型层"的语义，与方案 §14.1 的定位矛盾。

### 1.2 ARM 返回 `CPU`：会劫持真正的 CPU Backend

这一条更严重，是静默的功能破坏：

1. `ggml_backend_dev_by_type(CPU)` 返回**注册表中第一个** CPU 类型设备
   （`ggml-backend-reg.cpp:362-370`，线性扫描先命中即返回）；
2. FlagOS 的注册顺序在 ggml-cpu **之前**（`ggml-backend-reg.cpp:128` 的
   `ggml_backend_flagos_reg()` vs `179` 的 `ggml_backend_cpu_reg()`）；
3. 于是 `llama-context.cpp:353` 的
   `backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)`
   会拿到 FlagOS ARM Device；
4. `ggml_backend_sched_new()` 末位必须是 CPU 类型的断言
   （`ggml-backend.cpp:1789`）**照样通过**，问题不会在初始化阶段暴露；
5. 但真正的 CPU Backend 完全不在 sched 的 backend 列表里。所有
   `supports_op=false` 的节点在 `ggml_backend_sched_backend_id_from_cur()` 里无处可去，
   预分配张量路径直接撞 `ggml-backend.cpp:930` 的
   `GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation")`。

这与方案 §16.1 承诺的

```text
supports_op = false -> GGML Scheduler不把节点分给该FlagOS Device -> CPU或其他Backend执行
```

正好相反：Scheduler 已经没有 CPU 可选了。

同一劫持还会连带影响：

- `llama_numa_init()`（`llama.cpp:132-141`）会向 FlagOS ARM Device 的 reg 查询
  `ggml_backend_cpu_numa_init`，取不到即静默跳过 NUMA 初始化；
- `make_cpu_buft_list()`（`llama-model.cpp:937`）取 extra bufts 时同样查错设备，
  ggml-cpu 的 repack buft 全部丢失；
- `common/arg.cpp:1070-1071` 显式拒绝 CPU 类型设备
  （`if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) throw`），
  用户无法用 `-dev` 选中 ARM Provider。

### 1.3 建议

上游先例是 Hexagon：一块 DSP，`ggml-hexagon.cpp:3918` 直接
`return GGML_BACKEND_DEVICE_TYPE_GPU;`，唯一目的就是让它进入 `model->devices`
被正常调度。

- **SpacemiT K3**：返回 `GPU` 或 `IGPU`。`IGPU` 语义更贴合集成加速器，且
  `llama.cpp:255-275` 在没有独立 GPU 时会把 igpu 加入 `model->devices`；代价是有独显
  在场时会被丢弃，需按部署形态确认。
- **ARM Reference Provider**：返回 `ACCEL`，绝不返回 `CPU`。ACCEL 既不会劫持
  `dev_by_type(CPU)`，又能通过 §1.1 描述的 CPU buft list 首位拿到权重，符合
  Reference Provider 的定位。
- 真实硬件类别保留在 `flagos_device_caps.kind`（方案 §6.2 已有该字段）里，不外泄到
  GGML 的 device type。
- §5 line 155 的表述改为：**Device Type 按 llama.cpp 的调度语义选择，不按硬件形态
  描述**；并在 §13 和 §14 各加一句该 Provider 选定的类型及理由。

---

## 2. 必须修正：Graph UID 不是稳定的 Plan Cache Key

对应方案 §11.8：

> Plan Cache Key 优先使用稳定 Graph UID；没有 UID 时使用结构指纹。

`uid` 不稳定。`ggml_graph_next_uid()`（`ggml.c:56-69`）是单调递增计数器，而
`ggml_backend_sched_split_graph()` 在**每次调用**时都为顶层图（`ggml-backend.cpp:1074`）
和每个 split（`ggml-backend.cpp:1538`）重新取新值。因此同一张拓扑相同的图，每次重新
split 后 uid 都不同。

它对 CUDA Backend 有效（`ggml-cuda.cu:2555-2562`）的真正原因在上层：llama.cpp 在
`res->can_reuse(gparams)` 命中时会**整个跳过** `sched_reset()` 和
`sched_alloc_graph()`（`llama-context.cpp:1339-1352`），根本没有重新 split，于是上一次
的 uid 被原样重放，CUDA 才能据此断定"图逐字节未变、无需重新校验 node_props"。

按 uid 建 Plan Cache 的后果：每次图重建都 100% miss，Cache 无界增长，M3 的退出条件
"稳定图的 Plan 构建不再位于 Token 热路径"无法达成。

**建议**：把 §11.8 第 4 条反向表述 —— 结构指纹（Canonical Signature Hash）是唯一主键，
命中后仍需完整结构比较；`uid` 仅作为"与上次调用完全相同、可跳过结构校验"的 fast
path，且必须允许 `uid == 0`（`ggml_graph_view` 产生的图 uid 为 0，见 `ggml.c:7388-7404`
的初始化，`callback_eval` 路径下 `ggml-backend.cpp:1751` 会产生这种图）。

---

## 3. 必须修正：`graph_optimize` 中删除节点在结构上无效

对应方案 §11.9。自审方向正确，但结论偏软（"需要后续在 `graph_optimize` 或更上层完成
可验证的图改写"）。实际约束比这更硬。

`graph_optimize` 收到的是 `split->graph`，一个 `ggml_graph_view`
（`ggml-backend.cpp:1470` 调用点，`ggml.c:7388` 定义），其 `nodes` 指针**别名**原始
graph 的数组，`size == 0`。而紧随其后构建 `graph_copy` 的循环
（`ggml-backend.cpp:1490-1494`）是从**原始 graph 的 `[i_start, i_end)`** 区间重新抄写的：

```cpp
for (int j = split->i_start; j < split->i_end; j++) {
    sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
    graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
}
```

它不读 view 的 `n_nodes`。因此：

- 在 `graph_optimize` 里**原地置换节点顺序**有效（写入 view 的 `nodes` 即写入原数组，
  Metal / CUDA / Hexagon 正是这么用的）；
- **减少 `n_nodes`（删除节点）对 allocator 完全无效果** —— `graph_copy` 仍包含全部
  节点，`ggml_gallocr_alloc_graph_impl()`（`ggml-alloc.c:717-800`）仍会为每个中间张量
  计算生命周期并分配。

**建议**：§11.9 直接写明"在当前 Scheduler 代码路径下，`graph_optimize` 无法减少 GGML
分配"，避免后续反复尝试。真要降低分配峰值，只有两条路：改 llama.cpp 建图层（在
`build_graph` 阶段就不产生中间节点），或向上游提交 sched 改动。二者都不应放进 M3。

### 3.1 同时：§11.9 低估了 `graph_optimize` 的必要性

`ggml_can_fuse()`（`ggml-impl.h:699-712`）构造的是**连续下标**
`idxs[i] = node_idx + i`，`ggml_can_fuse_ext()`（`ggml-impl.h:669-696`）进一步要求
`node->src[0] 或 src[1] == 前一个节点`。也就是说邻接型融合要求候选节点在数组中**相邻**。
`graph_optimize` 是当前唯一能在不改语义的前提下制造这种邻接性的入口。

所以它不该被描述为"首版只允许保守重排或标记"的可选项，而是**邻接型融合 Pattern 的
前置依赖**，应在 M3 中与 Fusion Planner 同批交付。

---

## 4. 已有成果描述需更正：`RMS_NORM + MUL` 不是省带宽的融合

对应方案 §4 line 72（"`RMS_NORM + MUL` 真融合"）与 §23.6（作为第一个迁移 Pattern）。

看 Triton 源 `ggml/src/ggml-flagos/kernels/generate_flagos_kernels.py:483-500`：

```python
tl.store(norm_output + row * n_cols + cols, normalized, mask=mask)
tl.store(mul_output  + row * n_cols + cols, normalized * weights, mask=mask)
```

kernel 同时写 `norm_output` 和 `mul_output`，**两个中间张量都物化了**。C++ 侧
（`ggml-flagos.cpp:780-795`）同样把 `norm_dst->data` 和 `mul_dst->data` 都作为参数传入。

因此它省掉的是一次 kernel launch 和一次对 norm 结果的读，**写入字节数一点没减**。
把它当作"减少内存读写"的样板会得出错误的收益模型（方案 §11.5 的
`eliminated_memory_bytes` 项在这个 Pattern 上为 0）。

### 4.1 附带的正确性风险

当前实现（`ggml-flagos.cpp:1509-1516`）的融合判断只检查：

```cpp
const bool can_fuse = next && next->op == GGML_OP_MUL &&
    (next->src[0] == node || next->src[1] == node);
```

既没调用 `ggml_can_fuse()`，也没检查 norm 输出的 use count 和
`GGML_TENSOR_FLAG_OUTPUT`。它目前**安全**恰恰是因为两个输出都写了 —— norm 结果即使
还有别的消费者也读得到正确数据。

迁入公共 Pattern 后，如果为了省带宽改成只写 `mul_output`，这层意外保护就消失了，必须
同时接上 `ggml_can_fuse_subgraph_ext()`（`ggml.c:7622-7679`）的
use-count / `FLAG_OUTPUT` / view-src 检查。方案 §11.4 已列出这些约束，但需要在迁移条目
里显式标注"改写输出行为与接入合法性检查必须同批完成"。

**建议**：§4 的表述改为"`RMS_NORM + MUL` launch 合并（两个输出均物化）"；把它作为
M3 第一个迁移 Pattern 的理由改为"验证 Pattern Registry 与 Plan 通路"，而非"验证带宽
收益"，并明确其收益基线是 launch 数。

---

## 5. 跨 Provider Copy：当前代码给出了确切的失败模式

方案 §8.3 的 Copy 规则正确。补充一处 M1 必须同批修改的具体位置，否则是静默内存破坏。

`ggml_backend_buft_is_flagos()`（`ggml-flagos.cpp:1013-1015`）靠比较
`buft->iface.get_name` 函数指针判断归属：

```cpp
return buft != nullptr && buft->iface.get_name == ggml_backend_flagos_buffer_type_name;
```

多个 Provider 共享同一张 buffer-type vtable 后，`Denglin:0` 和 `ARM:0` 的 buft
**都会**通过这个检查。而 `ggml_backend_flagos_buffer_copy_tensor()`
（`ggml-flagos.cpp:1094-1112`）接下来只比较 `context->device` 这个 int：

```cpp
if (src_context->device != dst_context->device) { return false; }
...
cudaMemcpy(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice)
```

两个 `0` 相等，于是会对一个 host 指针发起 `cudaMemcpy D2D`。

同文件的 `supports_buft`（`ggml-flagos.cpp:2378-2382`）已经用了更强的判断
`ggml_backend_buft_is_flagos(buft) && buft->device == dev`，可以作为模板。

**建议**：M1 交付项中显式列出"`buft_is_flagos` 与 `buffer_copy_tensor` 改为按
Provider ID + Device UUID + Memory Domain 判断"，并加一条跨 Provider Copy 必须被拒绝
的单元测试（方案 §19.1 末条已提到，需落到 M1 而非 M5）。

---

## 6. 实施计划缺口：M0-M7 全程没有 decode 性能里程碑

M0 冻结基线、M1-M3 重构与融合、M4 上 ARM，全程没有一个针对 TG/decode 的性能交付项。
而 §4 的基线里 TG32 是 **17.28 vs CPU 16.97 token/s**，基本没有增益，§4.1 也已承认
"不能把 CUDA Graph 已经 Replay 等价为 Decode 已经优化完成"。

按此前在 KS20-A 上的实测（探针 `~/Denglin/work/tu-probe/fused_gemv.cu`，N=K=4096）：

| 路径 | 时间 | 备注 |
| --- | ---: | --- |
| 预先 dequant 成 F16 后走 GEMV | 0.840 ms | 另需一次 1.60 ms dequant，32 MiB F16 常驻 |
| 融合 dequant-GEMV，朴素访存 | 1.511 ms | 更慢 |
| **融合 dequant-GEMV，合并 uint32 访存** | **0.596 ms** | **1.41x**，仅 9 MiB |

决定性因素只是访存合并，把每元素整数除法提出内循环收益约 0%。这条收益与 Provider
抽象完全正交，不需要等 M1-M3。当前 decode 主路径（`ggml-flagos.cpp:1473-1499`，
`FLAGOS_DEQUANT_BLAS` 开启时走 `flagos_launch_mul_mat_dequant_blas`）正是被它击败的
那条，且 dequant 后的 F16 权重会常驻显存（`dequantized_weights` map，
`ggml-flagos.cpp:864-905`）。

**建议**：在 M0 与 M1 之间插入 **M0.5：decode 主路径换为融合 dequant-GEMV**，
交付真实 TG 提升与显存下降数据，然后再冻结基线去做结构重构。否则 M0 冻结的是一个
decode 无增益的基线，M3 结束时对外仍然拿不出 TG 数字，重构的价值也无法用端到端指标
证明（这与 §3.1 最后一条"用端到端指标验收"自相矛盾）。

---

## 7. 需要补强的细节

### 7.1 Manifest v3 的 `shape_constraints` 必须真正驱动 Selector

当前是 `format: 2`（`ggml/src/ggml-flagos/kernels/aot/manifest.json:2`），字段只有
`file / name / shared / num_warps / warp_size / block_size`。

关键约束：kernel 是 AOT 编译的，每个 `tl.constexpr`（`BLOCK`、`HAS_MASK` 等）在编译时
就被冻死，运行时没有重编译。因此 v3 新增的 `dtype_constraints` /
`shape_constraints`（方案 §10.1）必须真的被 Selector 用于**拒绝**，而不只是元数据。
否则 §19.3 的退出条件"不存在 `supports_op=true` 但执行选择为空"形同虚设 —— 现在这些
约束是硬编码在 `supports_op` 里的（如 `FLAGOS_RMS_NORM_MAX_COLS = 4096`，
`ggml-flagos.cpp:78`），迁移过程中很容易丢失。

建议在 M2 交付项中加一条：Manifest 约束到 `supports_op` 拒绝的**自动化一致性测试**，
覆盖每个已编译 Variant 的边界 shape。

### 7.2 ARM Provider 缺 Event 会连带禁掉多卡流水并行

方案 §13.2 计划"Queue：首版同步执行；Event：首版不提供"。

`llama-context.cpp:437-451` 在检查 pipeline parallelism 可行性时，只跳过 `CPU` 类型
backend，对其他所有类型都要求 `props.caps.async && props.caps.events`，任一缺失就
`pipeline_parallel = false` 并 break —— 关掉的是**整个模型**的流水并行。

若按 §1.3 建议 ARM 采用 `ACCEL` 类型进入 backend 列表，则一台同时插了多块 Denglin
卡的机器会因为 ARM Provider 在场而丢掉流水并行。

**建议**：要么 ARM Provider 提供真实的线程任务完成 Event（§13.2 已列为备选），要么在
文档中写明这一副作用并在 CMake 层默认不与 GPU Provider 同时启用。

### 7.3 `GGML_SCHED_MAX_BACKENDS = 16`

`ggml-backend.cpp:752-753`。四类 Provider 各枚举多设备时容易顶到上限，
`ggml-backend.cpp:1788` 只有一个裸 `GGML_ASSERT`。建议 M1 在 Device Registry 侧加上限
检查和可读的错误信息，并支持通过环境变量限制每个 Provider 枚举的设备数。

### 7.4 §5 的新 Device 命名与 M1 "行为不变"冲突

§5 line 149-153 提出 `FlagOS:Denglin:0` 形式的命名。当前实现是
`context->name = "FlagOS" + std::to_string(device)`（`ggml-flagos.cpp:2515`），
而 `ggml_backend_dev_by_name()` 做的是**精确匹配**（`ggml-backend-reg.cpp:352-359`，
`striequals`）。改名会直接破坏所有依赖 `-dev FlagOS0` 的脚本，包括
`ggml/src/ggml-flagos/validate_denglin.sh`。

这与 M1 退出条件"Denglin 全部测试和端到端基线通过 / 行为不变"冲突。
**建议**：M1 保留旧名作为 alias，正式改名放到 M2，并在 M2 交付项中列出需要同步更新的
脚本清单。

### 7.5 §7 的 Provider 静态注册需要一个空设备的降级路径

`ggml_backend_flagos_reg()`（`ggml-flagos.cpp:2477-2496`）在
`cudaGetDeviceCount` 失败时返回一个零设备的 reg，这是正确行为。多 Provider 后需要保证
同样语义：**任一 Provider 探测失败不影响其他 Provider 注册**，且全部失败时
`get_device_count` 返回 0 而不是抛异常 —— 否则在没有加速卡的开发机上会连
`llama_backend_init` 都过不去，正好破坏 §13.1 给 ARM Provider 设定的"无目标加速卡环境
中的可运行实现"定位。

---

## 8. 评审通过、建议保留的部分

以下结论经代码核对成立，属于方案中价值最高的部分：

- **§9.3 单一 Selector**。当前 `supports_op`（`ggml-flagos.cpp:1947+`）与执行
  `switch`（`ggml-flagos.cpp:1380+`）分别维护规则，已经出现过执行路径二次判断 shape
  的情况（如 `GGML_OP_MUL_MAT` 的 `columns` 选择，`ggml-flagos.cpp:1483-1487`）。
  统一是必要的。
- **§10.2 构建期生成 Typed Binding**，避免每 Token 热路径解析 JSON。
- **§11.4 复用 `ggml_can_fuse_subgraph_ext()`** 而不自写合法性检查。该函数
  （`ggml.c:7622-7679`）已经覆盖了 use-count 闭合、`FLAG_OUTPUT`、`FLAG_COMPUTE`
  和 view-src 链必须在子图内（weight 常量除外）这几个最容易出错的条件。
- **§12 structural / binding 双指纹**，且明确"Hash 命中后仍需比较 Canonical Graph"。
  这比当前 `flagos_graph_properties_changed()`（`ggml-flagos.cpp:1602-1624`）逐节点
  memcmp 全部 `ggml_tensor` 的做法更可扩展，同时保留了同等安全性。
- **§16.1 融合 Pattern 中每个计算节点都必须有独立执行路径**。这条直接挡住了
  Scheduler 逐节点询问 `supports_op`（`ggml-backend.cpp:884-889`）带来的真实风险：
  两个节点不保证一起进入同一 Backend。
- **§19.7 把"中间内存读写流量"与"GGML 分配峰值"分开记账**。配合 §3 的结论，这是唯一
  诚实的口径。
- **§2.8 首版不引入 `dlopen` 插件层**，接口保持 C-compatible。上游
  `ggml-backend-reg.cpp` 的 dl 加载路径（`register_backend` 带 `dl_handle_ptr`）已经
  存在，将来接入成本可控，现在不做是正确的克制。

---

## 9. 修正后建议的里程碑顺序

```text
M0    冻结当前 Denglin 基线（含融合关闭、Graph 关闭的对照）
M0.5  decode 主路径换为融合 dequant-GEMV，交付 TG 与显存数据   <- 新增
M1    抽 Provider 基础层，行为不变
        + buft_is_flagos / buffer_copy_tensor 按 Provider+UUID 判断
        + Device Type 决策落地（K3=GPU/IGPU，ARM=ACCEL）
        + MAX_BACKENDS 上限检查，Device 旧名保留 alias
M2    Signature / Catalog / Selector 统一，Manifest v3
        + Manifest 约束到 supports_op 拒绝的一致性测试
        + Device 正式改名 + 脚本清单更新
M3    graph_compute 内部 Plan Cache + 公共 Fusion Planner
        + graph_optimize 用于制造邻接性（不承诺降低分配）
        + RMSNorm+Mul 迁移，同批接入 can_fuse_subgraph_ext 检查
M4-M7 按原方案
```

Plan Cache Key 一律以结构指纹为主键，`uid` 仅作 fast path。
