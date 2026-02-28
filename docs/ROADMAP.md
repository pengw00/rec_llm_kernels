# Rec LLM Kernels — Roadmap

本仓库目标：构建一套“类似 vLLM 的推理 runtime（`rec_llm_runtime/`）”，底层使用自研/可替换的 CUDA kernels（`cuda/csrc/` + `rec_llm_kernels._C.ops`），重点聚焦 **Paged KV Cache +（Prefill/Decode）Attention + RMSNorm**，并最终跑通（至少）Llama 类 Transformer 的高吞吐推理。

---

## 0) 当前状态（2026-02-19）

- C++/CUDA kernels：`reshape_and_cache`、`rms_norm` 已实现；`flash_att_forward` 目前为占位实现（`out=q+k`）。
- 绑定层：算子导出在 `rec_llm_kernels._C.ops.*`。
- 上层 runtime：`runtime/src/rec_llm_runtime/engine/` 仍是空壳，尚未形成端到端闭环。
- 测试：`pytest` 套件已整理为可运行（无 CUDA 时自动 skip）。

---

## 1) 里程碑分期（从“能跑”到“可用/可快”）

### Phase 0 — 工程基线（1–2 天）
**目标**
- 在 Colab（T4/A100）稳定完成：`pip install -e .` + `pytest -q`
- 构建参数可控：自动/可覆盖 CUDA arch、可控并行编译（`MAX_JOBS`）

**验收**
- `pytest` 在有 CUDA + 已编译扩展环境下通过（或明确 skip 的项不阻塞）
- 失败时日志清晰（Torch/CMake/FlashInfer 依赖问题不再静默）

### Phase 1 — Kernel 闭环（2–5 天）
**目标**
- `rms_norm`、`reshape_and_cache` 正确性稳定（对齐 reference）
- attention 先以“占位/最简”方式打通调用链（或接入 FlashInfer decode）

**验收**
- `tests_cuda/test_rms_norm.py`、`tests_cuda/test_paged_cache.py` 稳定通过
- `tests_cuda/test_kernels.py` 明确验证当前占位 attention 语义

### Phase 2 — 最小推理 MVP（5–10 天）
**目标**
- 跑通一个最小 Transformer block 的 forward（建议先 decode-only）
- `rec_llm/engine` 形成“能跑”的最简闭环：分配 KV → 写 cache → attention → 输出

**建议交付**
- `rec_llm/engine/block_manager.py`：`BlockAllocator`（最小实现：分配/回收）
- `rec_llm/engine/cache_engine.py`：`CacheManager`（维护 `slot_mapping/block_tables/context_lens` 的最简版本）
- `rec_llm/engine/llm_engine.py`：`Engine.step()`（单 batch / 单步）

**验收**
- 新增 1 个端到端最小 demo（可以是小维度随机权重/配置），可在 Colab GPU 上运行

### Phase 3 — Paged Attention 正式化（10–20 天）
**目标**
- 分离并跑通两条路径：
  - Prefill（长 prompt）
  - Decode（高并发解码）
- attention 可以选择：
  - A) FlashInfer 集成（更快出效果）
  - B) 自研 kernel（更可控，但工作量更大）

**验收**
- attention 输出与 PyTorch reference 在可接受误差内一致
- 针对 prefill/decode 的正确性与形状约束有明确单测覆盖

### Phase 4 — 调度与吞吐（2–4 周）
**目标**
- 多请求并发（continuous batching）
- KV block 回收、OOM 保护、基础指标采集

**验收**
- 提供 benchmark 脚本/结果：tokens/s、p50/p95 latency、显存占用曲线

### Phase 5 — 性能优化与产品化（持续）
**目标**
- kernel 融合/向量化、减少同步与内存搬运、改进 cache layout
- wheel 打包、CI（build + unit tests）、release 文档

---

## 2) 模块边界（建议固定的接口）

### 2.1 `rec_llm_kernels._C.ops`（底层稳定 ABI）
建议把下面作为“稳定接口面”：
- `reshape_and_cache(key, value, key_cache, value_cache, slot_mapping) -> None`
- `rms_norm(out, x, weight, eps) -> None`
- `flash_att_forward(q, k, v, out) -> None`（当前占位；后续可替换为真实 attention）

### 2.2 `runtime/src/rec_llm_runtime/ops`（Python 稳定 API）
建议在 Python 侧做：
- dtype/shape 检查
- contiguous 处理
- fallback（可选）
- 文档化（输入输出约束）

### 2.3 `runtime/src/rec_llm_runtime/engine`（运行时核心）
建议职责：
- block 池管理、KV cache 元数据维护
- batch 组织与调度（后期）
- 给 `model_executor` 提供“已准备好的 cache view + 元数据”

### 2.4 `runtime/src/rec_llm_runtime/model_executor`（模型算子编排）
建议职责：
- 只关心“如何调用 ops/engine”，不关心分配策略细节

---

## 3) Colab 建议工作流（开发/验证）

```bash
# GPU runtime
nvidia-smi

# build (verbose) + tests
export MAX_JOBS=2
pip install -e cuda -v
pytest -q tests_cuda
```

如果是 T4（sm75），可显式：
```bash
export CMAKE_CUDA_ARCHITECTURES=75
```

---

## 4) 需要你确认的 3 个关键决策（决定 Phase 2/3 走向）

1. Attention 优先路线：FlashInfer 集成 / 自研 kernel？
2. 目标 GPU：T4（sm75）/ A100（sm80）？
3. MVP 优先：decode-only / prefill 优先？
