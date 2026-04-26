## 1. intro

### 1.1 为什么要引入 position embedding?

Self-Attention 本身只看 token 内容，不天然感知顺序。
如果不给位置信息，模型无法区分：
- “我喜欢你”
- “你喜欢我”

也就是说，attention 能建模“谁和谁相关”，但默认不知道“谁在前、谁在后”。
因此需要额外把位置信息注入进去。

### 1.2 绝对位置编码、相对位置编码与 RoPE 的区别
- **绝对位置编码**：直接告诉模型“这是第几个位置”。常见做法是把位置向量加到输入 embedding 上。
- **相对位置编码**：更关注两个 token 之间的距离关系，例如“我和你相隔 3 个位置”。
- **RoPE**：不把位置直接加到 token 表示上，而是把位置编码变成一个**旋转操作**，施加到 `Q/K` 上，使 attention score 自然带上相对位置信息。

### 1.3 RoPE 的核心
把位置编码变成一个二维旋转操作，直接作用在 `Q/K` 上。

---

## 2. RoPE

### 2.1 二维旋转的数学形式
设某一对维度组成二维向量：

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

若当前位置是 `m`，对应旋转角度为 `m\theta`，则旋转后：

$$
R(m\theta)\mathbf{x}=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

即：

$$
x'_1 = x_1\cos(m\theta) - x_2\sin(m\theta)
$$

$$
x'_2 = x_1\sin(m\theta) + x_2\cos(m\theta)
$$

本质上，RoPE 仍然是在编码绝对位置，只不过不是“直接相加”，而是“按位置旋转”。

### 2.2 为什么只作用在 Q/K 上？
attention score 的核心是：

$$
\text{score} = QK^T
$$

如果我们把位置信息编码进 `Q/K`，那么相似度计算时就会自然带上位置信息。
而 `V` 主要负责承载“被聚合的内容”，即我们要“取出来什么”，无需注入位置信息，通常不需要参与位置旋转。

### 2.3 从二维推广到高维
当 `head_dim` 很大时，可以自然地把它拆成多个二维子空间：
- `(0,1)` 一对
- `(2,3)` 一对
- `(4,5)` 一对
- ...

每一对维度对应一个旋转频率 `\theta_i`。
当位置为 `m` 时，第 `i` 对维度旋转 `m\theta_i`。

于是就得到：
- 不同维度对子空间旋转速度不同
- 低频分量变化慢，更偏全局
- 高频分量变化快，更偏局部

这也就是：**把高维空间拆成多个二维子空间后再分别旋转**。

---

## 3. YaRN

这里暂时只建立对原理的直观认识。

可以把 YaRN 理解为：**在长上下文外推时，对 RoPE 的频率分布和注意力缩放进一步做修正**，尽量缓解原始 RoPE 在超出训练长度后的位置失真问题。

一个便于记忆的直觉：
- **低频**：更像长周期变化，对全局结构更敏感
- **高频**：更像短周期变化，对细粒度局部位置更敏感

YaRN 的目标之一，就是在“长度变长”时，不让这些频率关系失真得太厉害，同时避免注意力分布过于发散。

---

## 4. source code

在 MiniMind 的官方实现中，RoPE 主要由两个函数完成：
- `precompute_freqs_cis`
- `apply_rotary_pos_emb`

### 4.1 `precompute_freqs_cis`
它的作用是：
1. 先为每个二维子空间计算频率
2. 再根据位置 `t` 得到每个位置的旋转角度
3. 最后得到 `cos/sin`

关键形式可以记为：

```python
freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
freqs = torch.outer(t, freqs)
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

可以配合其数学表达式理解（TODO）
此时：

```python
freqs_cos.shape == freqs_sin.shape == [seq_len, head_dim]
```

#### 为什么要 `cat` 两次？
因为后续希望直接写成：

```python
q_embed = q * cos + rotate_half(q) * sin
```

那么 `cos/sin` 的最后一维就必须和 `q` 的最后一维 `head_dim` 对齐，方便广播。

### 4.2 `apply_rotary_pos_emb`
其核心形式是：

```python
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

这里没有按照常规奇偶配对，而是通过 `rotate_half` 一次性完成了“二维旋转中那部分互换并带符号的操作”。

### 4.3 `rotate_half`
常见写法是：

```python
half = x.shape[-1] // 2
out = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
```

若：

```python
x = [a, b, c, d, e, f, g, h]
```

则：

```python
rotate_half(x) = [-e, -f, -g, -h, a, b, c, d]
```

它看起来不是“相邻两维一组”，而是“前半段/后半段一组”。
但和相邻奇偶配对完全等价，本质上均是：**把高维向量拆成多个二维子空间并旋转**。

---

## 5. 实现细节

### 5.1 `attn_factor`
`attn_factor` 是一个**缩放系数**，默认取 `1.0`。

### 5.2 `rotate_half()`
前文已经交代。

### 5.3 `unsqueeze_dim`
`cos/sin` 初始形状通常是：

```python
[seq_len, head_dim]
```

而 `q/k` 在多头场景下一般是：

```python
q: [batch_size, num_heads, seq_len, head_dim]
k: [batch_size, num_kv_heads, seq_len, head_dim]
```

因此要先通过 `unsqueeze`，让 `cos/sin` 能沿 batch/head 维广播。

例如：

```python
cos = cos.unsqueeze(0)
sin = sin.unsqueeze(0)
```

这样就能把：

```python
[seq_len, head_dim]
```

扩成：

```python
[1, seq_len, head_dim]
```

再进一步广播到 `q/k` 上。

一句话概括：

> `unsqueeze_dim` 的作用不是改变 RoPE 数学意义，而是为了让 `cos/sin` 在多头张量上正确广播。

### 5.4 GQA 中 RoPE 的插入位置
在 GQA 中，`q/k/v` 的头数不一定相同：
- `q` 是 `num_heads`
- `k/v` 是 `num_kv_heads`

因此在 apply RoPE 时，通常是：

```python
q/k reshape -> apply RoPE -> repeat_kv -> attention
```


---

## 6. Validation


### 6.1 数学直写版
先实现最朴素的版本：

```python
x_even = x[:, 0::2]
x_odd  = x[:, 1::2]
```

然后按二维旋转公式分别更新：

```python
x_rot_even = x_even * cos - x_odd * sin
x_rot_odd  = x_even * sin + x_odd * cos
```

这里可以和 RoPE 的数学公式对应。

### 6.2 MiniMind 实现

当把向量从 interleaved 排布重排到 half-split，再做 MiniMind 风格旋转，最后再转回去后，结果是：

```python
allclose = True
max abs diff = 0.
```

这说明：

> 数学直写版和 MiniMind 风格版本质完全等价，差别只在最后一维如何组织。

---
## 7. Questions
1. Self-Attention 为什么天然不感知顺序？如果完全不给位置信息，会出现什么问题？
2. RoPE 和“把位置向量直接加到 embedding 上”的做法，最本质的区别是什么？
3. 为什么 RoPE 主要作用在 `Q/K` 上，而不是作用在 `V` 上？
4. 从数学上看，RoPE 为什么可以理解成“把高维向量拆成多个二维子空间分别旋转”？
5. `precompute_freqs_cis` 生成的 `cos/sin` 最终为什么要和 `head_dim` 对齐？
6. `rotate_half` 为什么看起来不像“偶数位/奇数位配对旋转”？这和最后一维的排布方式有什么关系？
7. 在 MiniMind 的 GQA 实现里，RoPE 的插入位置为什么是在 `repeat_kv` 之前？

## 8. Conclusion

1. attention 本身不感知顺序，所以要引入位置编码
2. RoPE 不是把位置直接加到 embedding，而是把位置变成 `Q/K` 上的旋转，这是它和绝对位置编码的本质区别
3. RoPE 可以理解成把最后一维拆成多个二维子空间，再让每一对子空间按位置做旋转。
4. 工程实现上，可以写成：
   - 数学直写版：显式偶奇配对
   - 源码工程版：`rotate_half + cos/sin broadcast`
   - 两种写法表面不同，本质等价
5. 在 GQA 中，RoPE 的位置是：`q/k reshape` 后， `repeat_kv` 前
