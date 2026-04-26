## 1. Project intro
Day 2 的目标是围绕 MiniMind 中的 **Self-Attention** 建立一条更扎实的技术证据链，不只停留在“知道 attention 是什么”，而是进一步完成：
- 从 decoder-only LLM 角度理解 attention 的作用
- 梳理 attention 的核心张量 shape 变化
- 从单头 attention 手写实现，逐步扩展到标准多头 attention
- 在标准多头基础上，继续扩展到 MiniMind 实际采用的 **GQA（Grouped-Query Attention）**
- 为后续继续分析 RoPE、KV-cache 与推理加速路径打基础
`single-head attention -> standard multi-head attention -> MiniMind-style GQA `
---

## 2. Why Attention in decoder-only LLM

在 decoder-only LLM 中，当前 token 的表示不能只依赖它自己，还需要结合它左侧历史上下文的信息。  
Self-Attention 的作用就是：

**让当前位置的 token，动态地从历史 token 中选择“该关注谁、关注多少”，并把这些信息加权汇总成新的上下文表示。**

对于语言模型而言，这一点非常关键，因为生成当前 token 时，模型往往需要参考：

- 前文的实体与指代关系
    
- 句法结构是否闭合
    
- 主题是否延续
    
- 某些长距离依赖是否仍然成立
    

由于 MiniMind 是 **decoder-only** 结构，因此 attention 不是双向可见的，而必须配合 **causal mask** 使用。  
也就是说：

> 当前位置只能看当前位置及其左侧历史，不能偷看未来 token。

因此，attention 在 decoder-only LLM 中的本质可以概括为：

```text
当前 token
-> 与历史 token 计算相关性
-> 得到注意力分布
-> 对历史信息加权求和
-> 形成新的上下文表示
```

---

## 3. Attention computation chain

从计算流程上看，Self-Attention 可以抽象为：

```text
hidden_states
-> q_proj / k_proj / v_proj
-> attention scores
-> causal mask
-> softmax
-> weighted sum with v
-> output projection
```

若写成更具体的张量流，则为：

```text
x
-> q / k / v
-> scores = q @ k^T / sqrt(d)
-> scores + causal mask
-> attn = softmax(scores)
-> context = attn @ v
-> output
```

这条主线在单头、多头、GQA 三种实现中始终不变；变化的关键在于：

- q/k/v 的 shape 如何组织
    
- head 维度如何拆分
    
- K/V heads 是否与 Q heads 相同
    

---

## 4. Shape flow

## 4.1 single-head attention

在最小单头 attention 实现中，设：

- `B`：batch size
    
- `T`：sequence length
    
- `D`：hidden size
    

输入张量：

```text
x: [B, T, D]
```

经过线性映射后：

```text
q: [B, T, D]
k: [B, T, D]
v: [B, T, D]
```

然后计算：

```text
scores = q @ k^T / sqrt(D)
scores: [B, T, T]
```

这里 `scores` 的最后两个维度表示：

- 行：query 位置
    
- 列：key 位置
    

加入 causal mask 后做 softmax：

```text
attn: [B, T, T]
```

再与 `v` 相乘：

```text
context = attn @ v
context: [B, T, D]
```

最后输出：

```text
output = o_proj(context)
output: [B, T, D]
```

因此，单头 attention 的最核心 shape 流可以总结为：

```text
x       : [B, T, D]
q / k / v: [B, T, D]
scores  : [B, T, T]
attn    : [B, T, T]
context : [B, T, D]
output  : [B, T, D]
```

---

## 4.2 standard multi-head attention

单头 attention 的核心问题在于：  
所有关系都在同一个表示空间中建模，表达能力有限。

因此，标准多头 attention 的做法是：

> 将 hidden size 拆成多个 head，每个 head 在不同子空间中独立计算 attention。

设：

- `Nh = num_heads`
    
- `Dh = head_dim`
    
- `D = Nh * Dh`
    

则输入仍然为：

```text
x: [B, T, D]
```

经过投影后：

```text
q: [B, T, D]
k: [B, T, D]
v: [B, T, D]
```

然后做 reshape：

```text
q: [B, T, Nh, Dh]
k: [B, T, Nh, Dh]
v: [B, T, Nh, Dh]
```

再做 transpose，将 head 维提前：

```text
q: [B, Nh, T, Dh]
k: [B, Nh, T, Dh]
v: [B, Nh, T, Dh]
```

此时 attention score 为：

```text
scores = q @ k^T / sqrt(Dh)
scores: [B, Nh, T, T]
```

softmax 后：

```text
attn: [B, Nh, T, T]
```

再与 `v` 相乘得到每个 head 的输出：

```text
context: [B, Nh, T, Dh]
```

然后拼接 heads：

```text
context -> [B, T, Nh, Dh] -> [B, T, D]
```

最后：

```text
output = o_proj(context)
output: [B, T, D]
```

因此，标准多头 attention 的核心 shape 流为：

```text
x            : [B, T, D]
q/k/v        : [B, T, D]
split heads  : [B, T, Nh, Dh]
transpose    : [B, Nh, T, Dh]
scores       : [B, Nh, T, T]
attn         : [B, Nh, T, T]
context      : [B, Nh, T, Dh]
merge heads  : [B, T, D]
output       : [B, T, D]
```

---

## 4.3 MiniMind-style GQA
```text
x
-> q/k/v projection
-> split heads
-> repeat_kv
-> scores
-> causal mask
-> softmax
-> attn @ v
-> merge heads
-> output
```

在标准多头 attention 中，默认：

```text
Q heads = K heads = V heads
```

但 MiniMind 的 attention 实现采用 **GQA（Grouped-Query Attention）** 架构，即：

```text
Q heads != KV heads
```

也就是：

- Q 使用 `num_attention_heads`
    
- K/V 使用 `num_key_value_heads`
    


设：

- `Nh = num_attention_heads`
    
- `Nkv = num_key_value_heads`
    
- `Dh = head_dim = hidden_size // Nh`
    

则输入：

```text
x: [B, T, D]
```

投影后：

```text
q: [B, T, Nh * Dh]
k: [B, T, Nkv * Dh]
v: [B, T, Nkv * Dh]
```

reshape 后：

```text
q: [B, T, Nh, Dh]
k: [B, T, Nkv, Dh]
v: [B, T, Nkv, Dh]
```

由于 attention 计算时，Q/K/V 的 head 数需要对齐，因此 MiniMind 会通过 `repeat_kv` 将较少的 K/V heads 扩展到与 Q heads 一样多。

若：

```text
n_rep = Nh // Nkv
```

则：

```text
k: [B, T, Nkv, Dh] -> [B, T, Nh, Dh]
v: [B, T, Nkv, Dh] -> [B, T, Nh, Dh]
```

之后再 transpose：

```text
q: [B, Nh, T, Dh]
k: [B, Nh, T, Dh]
v: [B, Nh, T, Dh]
```

后续 attention 主线与标准多头完全一致：

```text
scores  : [B, Nh, T, T]
attn    : [B, Nh, T, T]
context : [B, Nh, T, Dh]
merge   : [B, T, D]
output  : [B, T, D]
```

因此，GQA 相比标准多头 attention，新增的关键点只有两个：

1. `k_proj / v_proj` 输出更少的 KV heads
    
2. 通过 `repeat_kv` 将 KV heads 扩展到与 Q heads 对齐
    

---

## 5. Why multi-head / GQA

### 5.1 why multi-head attention

多头 attention 的意义在于：

- 不同 head 可以在不同子空间中学习不同类型的依赖关系
    
- 模型不必把所有关系都挤在同一个表示空间中
    
- 更适合同时捕捉局部关系、长程依赖、句法边界、实体指代等信息
    

因此，多头 attention 的核心不是“复制很多份 attention”，而是：

> 将 hidden states 切分到不同 head 中，并行建模不同关系，再把这些信息拼接回去。

### 5.2 why GQA

GQA 的核心目的在于：

- 保留较多的 Q heads，维持 query 侧的表达能力
    
- 减少 K/V heads 的数量，从而降低参数量与缓存成本
    
- 在实际 attention 计算前，再通过 `repeat_kv` 完成对齐
    

因此，GQA 可以看作是：

> 在“保持 attention 主公式不变”的前提下，对 head 组织方式做工程优化。

---

## 6. Source code

围绕 `model/model_minimind.py` 展开，其中 attention 相关最关键的位置包括：

- `Attention`
    
- `Attention.__init__`
    
- `Attention.forward`
    
- `repeat_kv`
    

### 6.1 `Attention.__init__`

这一部分决定了：

- `num_attention_heads`
    
- `num_key_value_heads`
    
- `head_dim`
    
- `n_rep`
    

其中 GQA 的关键关系是：

```text
n_rep = num_attention_heads // num_key_value_heads
```

### 6.2 `Attention.forward`

这一部分负责 attention 的核心计算路径：

1. `q_proj / k_proj / v_proj`
    
2. reshape 成多头形式
    
3. 对 K/V 做 `repeat_kv`
    
4. transpose
    
5. 计算 scores
    
6. causal mask / softmax
    
7. `attn @ v`
    
8. merge heads
    
9. `o_proj`
    

### 6.3 `repeat_kv`

这是 MiniMind 中从标准多头 attention 进入 GQA 的关键函数。

它的作用是将较小的 KV heads 按组复制，从而扩展到与 Q heads 一致。
本质上仍是个 **shape 对齐函数**。

---

## 7. Implementation path

Day 2 的实现路径分为三步：

### 7.1 Step 1：single-head attention

先实现最小单头 attention，用于确认最核心的公式和 shape：

```text
x -> q/k/v -> scores -> mask -> softmax -> attn@v -> output
```


### 7.2 Step 2：standard multi-head attention

在单头代码基础上新增：

- `num_heads`
    
- `head_dim`
    
- split heads
    
- transpose
    
- merge heads
    

这一阶段的关键： 多头 attention 在单头 attention 外层增加了一个 head 维度。

### 7.3 Step 3：MiniMind-style GQA

在标准多头代码基础上继续新增：

- `num_key_value_heads`
    
- 更小的 `k_proj / v_proj`
    
- `repeat_kv`
    

这一阶段的关键：GQA 改变了 Q/K/V 的 head 组织方式。

---

## 8. Validation

本次测试的配置为：
```text
batch_size = 2
seq_len = 4
hidden_size = 8
num_heads = 2
num_key_value_heads = 1
hidden_dim = hidden_size // num_heads
n_rep = num_heads // num_key_value_heads
```
实际 forward 运行结果为：
```bash
Input x: torch.Size([2, 4, 8])

q/k/v shape:
q: torch.Size([2, 4, 8])
k: torch.Size([2, 4, 4]) # [B, T, num_key_value_heads * head_dim]
v: torch.Size([2, 4, 4])

after repeat k/v shape:
k: torch.Size([2, 2, 4, 4])
v: torch.Size([2, 2, 4, 4])

scores shape:
torch.Size([2, 2, 4, 4]) # [B, num_heads, T, T]

attn shape:
torch.Size([2, 2, 4, 4])

attn row sum:
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.]]])

output shape:
torch.Size([2, 4, 8]) # [B, T, D]

final output shape:
torch.Size([2, 4, 8])
```

---

## 9. Questions
这里列出一些问题，以检验学习并鼓励复习。

### Q 1：decoder-only 中的 attention 和 encoder 的 attention 有什么关键区别？

A：decoder-only attention 使用 causal mask，当前位置只能看到历史和当前，不能看到未来；而 encoder 的双向 attention 一般没有这个限制。

### Q 2：为什么多头 attention 不是简单复制多个单头？

A：因为多头 attention 会先把 hidden size 切分到多个子空间中，每个 head 在自己的子空间中独立建模关系，最后再拼接回来，因此本质是“分块并行建模”，不是简单复制。

### Q 3：GQA 和标准多头 attention 的核心区别是什么？

A：标准多头中 Q/K/V 的 head 数通常相同；GQA 中 Q heads 保持较多，而 K/V heads 可以更少，之后再通过 `repeat_kv` 扩展到与 Q heads 对齐。

### Q 4：MiniMind 中 `repeat_kv` 的作用是什么？

A：它将 shape 为 `[B, T, num_key_value_heads, head_dim]` 的 K/V 张量沿 head 维复制为 `[B, T, num_attention_heads, head_dim]`，从而使 attention 计算时 Q/K/V 的 head 数一致。

### Q 5：什么时候 GQA 会退化成标准多头 attention？

A：当 `num_key_value_heads == num_attention_heads` 时，`n_rep = 1`，`repeat_kv` 不做任何复制，此时就退化成标准 MHA。

---

## 11. Conclusion

1. 理解注意力机制从单头到多头再到 GQA 的清晰演进路径
    
2. 能够解释 attention 的核心张量每一步的变化
    
3. 已经能把 MiniMind 中 attention 的实现分成三层来理解：
    
    - 单头 attention
        
    - 标准多头 attention：head 维度拆分与拼接
        
    - GQA：Q/KV heads 解耦与 `repeat_kv`
        