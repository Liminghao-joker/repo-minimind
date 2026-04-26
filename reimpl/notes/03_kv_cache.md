# KV-cache for inference acceleration

## Project Intro

**Today Goal**  
围绕 MiniMind 当前源码，理解并验证 KV-cache 如何服务 decoder-only LLM 的推理加速，重点搞清楚它的动机、prefill / decode 两阶段、cache 的 shape 组织方式，以及在源码中的构造、更新与复用路径。

**Today Outcome**
- 明确了 KV-cache 的核心作用：缓存历史 token 的 K/V，避免 autoregressive generation 中对整段历史的重复计算。
- 顺着 MiniMind 源码梳理了 cache 的组织形式、更新逻辑，以及 `start_pos` 与 RoPE 的关联。
- 完成了两类最小验证思路：一类验证 cache 长度增长，一类比较开启 / 关闭 cache 时的端到端生成耗时；其中 `use_cache=True` 相比 `use_cache=False` 的整体生成耗时约提升 **1.43 x**。

---

## Core Problem

在 autoregressive generation 中，模型一次只生成一个新 token。问题在于：**新 token 需要关注全部历史 token，但历史 token 的表示在上一轮其实已经算过了**。如果不做任何缓存，那么每生成一步，都要把“历史序列 + 当前新 token”重新送进模型，再完整做一遍 q/k/v 投影、RoPE、attention 和后续层前向。

KV-cache 解决的就是这个问题：

> 既然历史 token 的 K/V 在这一层已经确定，那么后续 decode 时就没有必要再重复计算它们；只需要为当前新 token 计算一次新的 q/k/v，再把新的 K/V 追加到已有缓存上即可。

因此，KV-cache 的价值不在于改变模型输出逻辑，而在于**改变推理时的计算组织方式**。

---

## Intuition

可以把 decode 阶段理解成一种“状态递推”。

- prefill 时，完整 prompt 被一次性处理，并在每一层留下历史 K/V 状态；
- decode 时，每次只输入 1 个新 token，让它拿自己的 Q 去查询“历史 K/V + 当前步新产生的 K/V”；
- 历史本身仍然参与注意力，但历史 K/V 不再被重复投影和重复保存。

---

## Principle

### 1. Main idea

KV-cache 的作用，是把自回归生成中“历史 token 的 K/V 投影结果”保存下来，让 decode 阶段每次只为新 token 计算一次 Q/K/V，并用“新 Q + 全历史 K/V”做注意力，从而避免对整段历史反复做 K/V 投影与重复 attention 计算。

进一步说，新增 token 只需要拿自己的 `Q` 去查询“已有历史 K/V + 当前步的 K/V”，即：历史 K/V 会被复用，历史 Q 不会被复用。

### 2. Key variables
- `past_key_values`：按层保存的历史 cache，列表长度等于 `num_hidden_layers`。
- `past_key_values[i]`：第 `i` 层 block 的缓存，结构为 `(K_cache, V_cache)`。
- `past_key_values[0][0].shape[1]` ：历史长度 `Lpast` 。
- `start_pos`：当前输入 token 在整段序列中的起始位置编号，用于 RoPE 切片。
- `presents`：当前轮 forward 之后，每一层新生成的 cache 列表，下一轮会继续作为 `past_key_values` 使用。

### 3. Mechanism
1. prefill 阶段输入完整 prompt，建立每一层的初始 `(K_cache, V_cache)`。
2. decode 阶段每次只输入新增 token，并读取旧的 `past_key_values`。
3. 当前步新产生的 `xk / xv` 与旧 cache 沿时间维拼接，形成更长的 cache。
4. 各层将新的 `present` 汇总为 `presents`，供下一轮继续复用，从而实现增量生成。

---

## Math View

### 1. Formula

$\text{attn}(q_t, K_{\le t}, V_{\le t})$

解释：  
- `q_t`：当前新 token 在某一层的 query；
- `K_{\le t}, V_{\le t}`：截至当前位置为止，全部历史 token 以及当前 token 的 key / value；
- 这个表达式本质上说明：**未来只需要当前步的 Q，但需要全部历史的 K/V**。
---

## Shape Flow

下面统一记号：
- `B`: batch size
- `T`: 当前这一次 forward 的输入长度
- `Lpast`: 历史 cache 长度
- `hq`: query heads 数
- `hkv`: key/value heads 数
- `d`: head_dim

### 1. Prefill stage
- Input hidden states: `[B, T, H]`
- After `q_proj`: `[B, T, hq * d]`
- After `k_proj / v_proj`: `[B, T, hkv * d]`
- After `view`:  
  - `xq -> [B, T, hq, d]`  
  - `xk -> [B, T, hkv, d]`  
  - `xv -> [B, T, hkv, d]`
- After cache save:  
  - `K_cache -> [B, T, hkv, d]`  
  - `V_cache -> [B, T, hkv, d]`

### 2. Decode stage
若旧 cache 长度为 `Lpast`，当前新输入长度为 `Tnew`（通常为 1）：
- `past_key_value[0].shape = [B, Lpast, hkv, d]`
- `xk.shape = [B, Tnew, hkv, d]`
- `xv.shape = [B, Tnew, hkv, d]`
- After concat:
  - `xk -> [B, Lpast + Tnew, hkv, d]`
  - `xv -> [B, Lpast + Tnew, hkv, d]`

可以看到，当 decode 阶段每输入一个新的 token 时，KV-Cache 便是在 sequence length 的维度上进行拼接。

### 3. Meaning of `past_key_values[0][0].shape[1]`
`past_key_values[0][0]` 表示第 0 层的 `K_cache`，其 shape 为：

$
[B,\ L_{past},\ num\_key\_value\_heads,\ head\_dim]
$

因此：

```python
past_key_values[0][0].shape[1]
```

取到的正是 `Lpast`，即**当前已经缓存了多少个历史 token**。

---

## Code Path
### 1. Initialize cache slots
初始化每层的 cache 槽位：
```python
past_key_values = past_key_values or [None] * len(self.layers)
```


### 2. Compute current start position
读取历史长度，用于后续的 RoPE：

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
```

### 3. Slice RoPE by history length
根据历史长度，切 RoPE：

```python
position_embeddings = (
    self.freqs_cos[start_pos:start_pos + seq_length],
    self.freqs_sin[start_pos:start_pos + seq_length]
)
```

注意：decode 阶段的输入，它的 RoPE 位置不能重新从 0 开始，而要从 `start_pos` 继续往后接。

### 4. Append new K/V to old cache
把当前的 kv-cache 接到旧 cache 后面，`dim=1` 说明沿着时间维进行拼接：

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```


### 5. Collect per-layer presents

```python
presents = []
for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
    hidden_states, present = layer(...)
    presents.append(present)
```

每一层都会返回自己更新后的 `(K, V)`，最后汇总成新的 `presents`，作为下一轮 decode 的 `past_key_values`。

---
## Verification

### 1. Minimal checks
- 检查 prefill 后每层 `K_cache / V_cache` 的长度是否等于 prompt 长度。
- 检查 decode 一步后每层 cache 长度是否恰好 `+1`。
- 检查 decode 阶段当前输入是否可以缩到 1 个新 token。
- 对比 `use_cache=False` 与 `use_cache=True` 的端到端生成耗时。

### 2. Example assertions

```python
# prefill 后 cache 长度等于 prompt 长度
assert k.shape[1] == prompt_len

# decode 一步后 cache 长度 +1
assert new_k.shape[1] == old_k.shape[1] + 1

# 端到端 benchmark
print(f"use_cache=False: {t_no_cache:.4f}s")
print(f"use_cache=True : {t_cache:.4f}s")
```

### 3. Experiment notes

**Experiment 1: cache growth check**  
通过手动两步 forward，可以验证：prefill 结束后，cache 长度等于当前 prompt 长度；decode 一步后，cache 长度增长 1。

**Experiment 2: end-to-end generation benchmark** 
在相同 prompt 和生成长度下，比较 `generate()` 的总耗时：
```bash
use_cache=False: 1.0123s, output shape=(1, 256)
use_cache=True : 0.7079s, output shape=(1, 256)
speedup = 1.43x
```
这个实验测得的是**完整生成流程的端到端耗时**，不是某一层 attention 的纯算子速度。它说明 KV-cache 在真实 autoregressive generation 中确实带来了整体推理收益。

---

## Key Takeaways

- KV-cache 的本质是“历史 K/V 的状态复用”，而不是改变 attention  公式本身。
- prefill 和 decode 是理解 KV-cache 最关键的两阶段：前者一次性处理完整 prompt，并建立每层初始 cache，后者为新增 token 计算当前步 `Q/K/V` , 追加 cache。
- `past_key_values[0][0].shape[1]` 本质上就是历史长度 `Lpast` (当前已经缓存了多少历史 token），它同时决定了 cache 拼接位置和 RoPE 的起始位置。
---

## Review Questions

1. KV-cache 试图消除的重复计算，具体发生在 autoregressive generation 的哪个阶段？
2. 为什么通常只缓存历史 `K/V`，而不缓存历史 `Q`？
3. prefill 和 decode 的区别是什么？两者在输入长度和 cache 状态上分别有什么特点？
4. `past_key_values[i]` 在 MiniMind 里保存的是什么？它的基本结构应该怎么描述？
5. `past_key_values[0][0].shape[1]` 为什么可以直接理解成历史长度 `Lpast`？
6. `torch.cat([past_k, k], dim=1)` 里的 `dim=1` 为什么表示沿时间维追加？如果拼错维度会出现什么问题？
7. `start_pos` 为什么不能省略？它和 RoPE 的位置切片是什么关系？
8. GQA 下 cache 存的 `K/V` 头数和最终参与 attention 的头数为什么可能不同？
