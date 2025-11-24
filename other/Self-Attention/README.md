# Self-Attention（自注意力机制）

本目录包含了自注意力机制（Self-Attention）的详细理论讲解、代码实现和可视化演示。

## 📁 目录结构

```
Self-Attention/
├── 1.math.md              # 自注意力机制数学原理详解
├── 2.multi-headed.md      # 多头注意力机制详解
├── 3.ResidualConnection.md # 残差连接
├── 4.encoder.md           # 编码器结构
├── 5.decoder.md           # 解码器结构
├── 6.CausalMask.md        # 因果掩码
├── code/                  # 代码实现
│   ├── attention_simple.py    # 简洁版注意力实现
│   ├── attention_complete.py  # 完整版注意力实现
│   ├── attention_workflow.py  # 工作流程演示
│   └── attention_mechanism/   # 可视化输出
└── images/                # 图片资源
    └── multi-head.png     # 多头注意力示意图
```

## 📚 学习路径

### 1. 理论基础
- **[1.math.md](./1.math.md)** - 从数学角度深入理解自注意力机制
  - Q、K、V 矩阵的含义与计算
  - 维度变换的数学原理（512→64→512）
  - 注意力权重的计算过程
  - 位置编码、FFN、Mask 等关键概念

- **[2.multi-headed.md](./2.multi-headed.md)** - 多头注意力机制详解
  - 为什么需要多头注意力
  - 多头并行的计算过程
  - 常见误解与正确理解
  - 生动的比喻解释

### 2. 架构组件
- **[3.ResidualConnection.md](./3.ResidualConnection.md)** - 残差连接与层归一化
- **[4.encoder.md](./4.encoder.md)** - Transformer 编码器结构
- **[5.decoder.md](./5.decoder.md)** - Transformer 解码器结构
- **[6.CausalMask.md](./6.CausalMask.md)** - 因果掩码的实现与作用

### 3. 代码实现
#### 简洁版实现
```bash
cd code/
python attention_simple.py
```
- 生成 `attention_mechanism/attention_simple.html`
- 提供直观的多头注意力可视化界面
- 可切换查看不同注意力头的计算过程

#### 完整版实现
```bash
cd code/
python attention_complete.py
```
- 实现完整的自注意力机制
- 包含详细的数学计算步骤
- 适合深入理解实现细节

#### 工作流程演示
```bash
cd code/
python attention_workflow.py
```
- 展示自注意力机制的完整工作流程
- 从输入到输出的每一步变换
- 适合理解整体架构

## 🎯 核心概念速览

### 自注意力机制的核心公式
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 多头注意力
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 维度变换（标准Transformer）
- **输入嵌入**: 512维
- **每个头**: 64维（8个头）
- **FFN隐藏层**: 2048维
- **输出**: 512维

## 🚀 快速开始

1. **理解数学原理**：从 `1.math.md` 开始，建立扎实的数学基础
2. **学习多头机制**：阅读 `2.multi-headed.md`，理解为什么需要多头
3. **运行可视化**：执行 `code/attention_simple.py`，直观感受计算过程
4. **深入代码**：研究 `code/attention_complete.py`，理解实现细节

## 📖 推荐学习顺序

1. 先读 `1.math.md` 建立数学直觉
2. 再看 `2.multi-headed.md` 理解多头注意力
3. 运行 `attention_simple.py` 可视化代码
4. 阅读 `3-6.md` 了解完整架构
5. 研究 `attention_complete.py` 深入实现

## 🎨 可视化特色

- **交互式界面**：可切换查看不同注意力头
- **矩阵热图**：直观展示权重分布
- **完整流程**：从输入到输出的每一步变换
- **数学对应**：代码与理论公式一一对应

## 📊 代码特点

- **简洁版**：核心逻辑清晰，适合初学者
- **完整版**：包含所有细节，适合深入研究
- **工作流版**：展示完整数据处理流程
- **可视化**：HTML交互界面，直观展示计算过程

## 💡 学习提示

1. **不要死记公式**：理解每个符号的物理意义
2. **多看可视化**：直观感受矩阵变换过程
3. **动手实验**：修改参数观察结果变化
4. **对比学习**：比较单头与多头的差异

---

*本目录提供了从理论到实践的完整学习路径，帮助您深入理解自注意力机制这一现代深度学习的核心组件。*