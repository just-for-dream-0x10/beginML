# PyTorch NLP 核心笔记

## 引言

PyTorch 因其灵活性和动态图机制，在学术界和工业界的 NLP 领域得到了广泛应用。本笔记旨在梳理使用 PyTorch 进行 NLP 任务开发时的核心知识点。

## 一、文本表示 (Text Representation)

计算机无法直接理解文本，因此需要将文本转化为数值表示。

### 1. 词嵌入 (Word Embeddings)

词嵌入是将离散的词语映射到低维连续向量空间的技术，使得语义相近的词在向量空间中也相近。

*   **底层原理：** 基于分布假说——词的含义由其上下文决定。通过学习词语与其上下文之间的关系来得到词向量。
*   **模型设计：**
    *   **Word2Vec (Mikolov et al., 2013)**
        *   **CBOW (Continuous Bag-of-Words):** 根据上下文词预测中心词。
            *   目标函数 (简化版): 最大化给定上下文时中心词的条件概率。
            *   $J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$
        *   **Skip-gram:** 根据中心词预测上下文词。
            *   目标函数 (简化版): 最大化给定中心词时上下文词的条件概率。
            *   $J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t)$
            *   常用负采样 (Negative Sampling) 来优化计算效率。
    *   **GloVe (Pennington et al., 2014)**
        *   原理：利用全局词共现统计信息。
        *   目标函数: $J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$
            *   $w_i$: 中心词向量, $\tilde{w}_j$: 上下文词向量, $X_{ij}$: 词 $i$ 和词 $j$ 的共现次数, $f(X_{ij})$: 权重函数。
    *   **FastText (Bojanowski et al., 2017)**
        *   原理：将词看作字符n-gram的集合，为每个n-gram学习向量表示，词向量是其n-gram向量的和。能更好地处理未登录词 (OOV)。
*   **PyTorch 实现:**
    *   `torch.nn.Embedding(num_embeddings, embedding_dim)`
    *   `num_embeddings`: 词典大小。
    *   `embedding_dim`: 词向量维度。
    *   这个模块本质上是一个查找表，权重矩阵 $W \in \mathbb{R}^{\text{vocab_size} \times \text{embedding_dim}}$。输入一个词的索引，输出对应的词向量。
    *   可以通过 `padding_idx` 参数指定填充词的索引，使其向量在训练过程中不更新（通常为零向量）。
    *   可以加载预训练的词向量 (如 Word2Vec, GloVe) 来初始化 `Embedding` 层的权重。

    ```python
    import torch
    import torch.nn as nn

    # 假设词典大小为 10000，嵌入维度为 300
    vocab_size = 10000
    embedding_dim = 300
    embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    # 假设输入是一个包含词索引的序列 (batch_size=2, seq_len=5)
    input_indices = torch.tensor([,], dtype=torch.long)
    embedded_vectors = embedding_layer(input_indices)
    # embedded_vectors.shape will be (2, 5, 300)
    print(embedded_vectors.shape)
    ```

## 二、核心序列模型

### 1. 循环神经网络 (Recurrent Neural Networks, RNN)

RNN 专门用于处理序列数据，通过循环结构来捕捉序列中的时间依赖性。

*   **底层原理：** 当前时间步的隐藏状态 $h_t$ 由当前时间步的输入 $x_t$ 和前一时间步的隐藏状态 $h_{t-1}$ 共同决定。
*   **数学公式 (Simple RNN):**
    *   $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
    *   $y_t = W_{hy}h_t + b_y$
    *   $W_{hh}, W_{xh}, W_{hy}$ 是权重矩阵, $b_h, b_y$ 是偏置项, $\tanh$ 是激活函数。
*   **模型设计：**
    *   可以堆叠多层 RNN (Stacked RNN) 以学习更复杂的表示。
    *   可以是单向或双向 (Bidirectional RNN)。双向RNN能同时考虑过去和未来的上下文信息。
*   **问题：**
    *   **梯度消失 (Vanishing Gradients):** 在长序列中，梯度在反向传播时可能指数级衰减，导致模型难以学习长期依赖。
    *   **梯度爆炸 (Exploding Gradients):** 梯度指数级增长，导致训练不稳定。梯度裁剪 (Gradient Clipping) 是常用的缓解方法。
*   **PyTorch 实现:** `torch.nn.RNN`

    ```python
    # 输入维度 50, 隐藏层维度 100, 2层RNN
    input_size = 50
    hidden_size = 100
    num_layers = 2
    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    # batch_first=True 表示输入的维度顺序是 (batch, seq, feature)

    # 假设输入 (batch=3, seq_len=10, feature_dim=50)
    input_seq = torch.randn(3, 10, input_size)
    # 初始隐藏状态 (num_layers * num_directions, batch, hidden_size)
    h0 = torch.randn(num_layers * 2, 3, hidden_size) # *2 for bidirectional
    output, hn = rnn(input_seq, h0)
    # output.shape: (3, 10, hidden_size * 2)
    # hn.shape: (num_layers * 2, 3, hidden_size)
    print(output.shape, hn.shape)
    ```

### 2. 长短期记忆网络 (Long Short-Term Memory, LSTM)

LSTM 是一种特殊的 RNN，通过引入门控机制来解决梯度消失问题，从而更好地捕捉长期依赖。

*   **底层原理：** 引入三个门（遗忘门、输入门、输出门）和一个细胞状态 (Cell State) $C_t$ 来控制信息的流动和保留。
*   **数学公式：**
    *   遗忘门 (Forget Gate): $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$  (决定从细胞状态中丢弃什么信息)
    *   输入门 (Input Gate): $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ (决定让哪些新信息存储到细胞状态中)
    *   候选细胞状态 (Candidate Cell State): $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
    *   细胞状态更新 (Cell State Update): $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ (丢弃部分旧信息，加入部分新信息)
    *   输出门 (Output Gate): $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ (决定输出细胞状态的哪一部分)
    *   隐藏状态 (Hidden State): $h_t = o_t \odot \tanh(C_t)$
    *   $\sigma$ 是 Sigmoid 函数，$\odot$ 表示逐元素乘积。
*   **模型设计：** 类似 RNN，可堆叠、双向。
*   **PyTorch 实现:** `torch.nn.LSTM` (参数与 `nn.RNN` 类似，但返回 `output, (hn, cn)`)

    ```python
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    # 初始隐藏状态和细胞状态
    h0 = torch.randn(num_layers * 2, 3, hidden_size)
    c0 = torch.randn(num_layers * 2, 3, hidden_size)
    output, (hn, cn) = lstm(input_seq, (h0, c0))
    print(output.shape, hn.shape, cn.shape)
    ```

### 3. 门控循环单元 (Gated Recurrent Unit, GRU)

GRU 是 LSTM 的一种变体，结构更简单，参数更少，计算效率通常更高，性能与 LSTM 相当。

*   **底层原理：** 将 LSTM 的遗忘门和输入门合并为更新门 (Update Gate)，并将细胞状态和隐藏状态合并。
*   **数学公式：**
    *   重置门 (Reset Gate): $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
    *   更新门 (Update Gate): $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
    *   候选隐藏状态 (Candidate Hidden State): $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$
    *   隐藏状态 (Hidden State): $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
*   **PyTorch 实现:** `torch.nn.GRU` (参数与 `nn.RNN` 类似，返回 `output, hn`)

    ```python
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    h0 = torch.randn(num_layers * 2, 3, hidden_size)
    output, hn = gru(input_seq, h0)
    print(output.shape, hn.shape)
    ```

### 4. Transformer (Vaswani et al., 2017 "Attention Is All You Need")

Transformer 模型完全基于自注意力机制 (Self-Attention Mechanism)，摒弃了 RNN 的循环结构，从而实现了更好的并行化，并在许多 NLP 任务上取得了 SOTA 效果。

*   **底层原理：**
    *   **自注意力机制 (Self-Attention):** 计算序列中每个词与其他所有词之间的关联度（权重），然后加权求和得到该词的新的表示。
        *   **Scaled Dot-Product Attention:**
            $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
            *   $Q$ (Query), $K$ (Key), $V$ (Value) 是输入序列经过不同线性变换得到的矩阵。
            *   $d_k$ 是 Key 向量的维度，用于缩放，防止点积结果过大导致 softmax 进入梯度饱和区。
    *   **多头注意力 (Multi-Head Attention):** 将 $Q, K, V$ 投影到多个不同的子空间中分别进行注意力计算，然后将结果拼接并再次线性变换。这允许模型在不同表示子空间中关注来自不同位置的信息。
        $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$
        where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
    *   **位置编码 (Positional Encoding):** 由于 Transformer 没有循环结构，无法捕捉序列顺序信息，因此需要向输入嵌入中加入位置编码。通常使用正弦和余弦函数：
        $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
        $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
        *   $pos$ 是位置索引, $i$ 是维度索引, $d_{model}$ 是嵌入维度。
*   **模型设计 (Encoder-Decoder 架构):**
    *   **Encoder:** 由 N 个相同的层堆叠而成，每层包含一个多头自注意力子层和一个逐位置前馈神经网络 (Position-wise Feed-Forward Network) 子层。每个子层后都有残差连接 (Residual Connection) 和层归一化 (Layer Normalization)。
    *   **Decoder:** 也由 N 个相同的层堆叠而成，每层在 Encoder 的两个子层基础上，额外插入一个多头注意力子层，用于关注 Encoder 的输出 (Encoder-Decoder Attention)。Decoder 的自注意力层需要使用掩码 (Masking) 来防止当前位置关注到未来位置的信息（在序列生成任务中）。
*   **PyTorch 实现:**
    *   `torch.nn.Transformer` (完整的 Encoder-Decoder 模型)
    *   `torch.nn.TransformerEncoder`, `torch.nn.TransformerDecoder`
    *   `torch.nn.TransformerEncoderLayer`, `torch.nn.TransformerDecoderLayer`
    *   `torch.nn.MultiheadAttention`

    ```python
    # 示例：使用 TransformerEncoderLayer
    d_model = 512  # 模型的维度 (embedding_dim)
    nhead = 8      # 多头注意力的头数
    num_encoder_layers = 6
    dim_feedforward = 2048 # 前馈网络的隐藏层维度
    dropout = 0.1

    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    # 假设输入 (batch=3, seq_len=10, feature_dim=d_model)
    src = torch.rand(3, 10, d_model)
    # 可选的 src_key_padding_mask 用于指示哪些是padding token
    # src_key_padding_mask = torch.tensor([[False, False, False, True, True], ...], dtype=torch.bool)
    output = transformer_encoder(src) #, src_key_padding_mask=src_key_padding_mask)
    # output.shape: (3, 10, d_model)
    print(output.shape)

    # 对于 MultiheadAttention 的直接使用
    multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
    # 假设 Q, K, V 都是 src
    attn_output, attn_output_weights = multihead_attn(src, src, src)
    # attn_output.shape: (3, 10, d_model)
    # attn_output_weights.shape: (3, 10, 10) (每个query对所有key的注意力权重)
    print(attn_output.shape, attn_output_weights.shape)
    ```

## 三、训练规则

### 1. 损失函数 (Loss Functions)

损失函数衡量模型预测与真实标签之间的差异。

*   **交叉熵损失 (Cross-Entropy Loss):** 常用于分类任务（如文本分类、序列标注的每个时间步分类）。
    *   公式 (对于单个样本，多分类): $L = -\sum_{c=1}^{M} y_c \log(p_c)$
        *   $M$ 是类别数, $y_c$ 是指示变量 (如果真实类别是 $c$ 则为1，否则为0), $p_c$ 是模型预测类别为 $c$ 的概率。
    *   **PyTorch 实现:**
        *   `torch.nn.CrossEntropyLoss`: 内部包含了 `LogSoftmax` 和 `NLLLoss`。输入为原始 logits。
        *   `torch.nn.NLLLoss` (Negative Log Likelihood Loss): 输入为 log-probabilities。
        *   `torch.nn.BCELoss` (Binary Cross-Entropy Loss): 用于二分类任务。
        *   `torch.nn.BCEWithLogitsLoss`: 结合了 Sigmoid 和 BCELoss，数值更稳定。

    ```python
    criterion = nn.CrossEntropyLoss()
    # 假设模型输出 logits (batch_size=4, num_classes=3)
    outputs = torch.randn(4, 3, requires_grad=True)
    # 真实标签 (batch_size=4)
    labels = torch.tensor(, dtype=torch.long)
    loss = criterion(outputs, labels)
    print(loss)
    ```
*   **其他损失：**
    *   **均方误差损失 (Mean Squared Error Loss):** `torch.nn.MSELoss` (常用于回归任务)。
    *   **CTC Loss (Connectionist Temporal Classification):** 用于序列标注任务，其中对齐是可变的（如语音识别）。`torch.nn.CTCLoss`。

### 2. 优化器 (Optimizers)

优化器根据损失函数计算得到的梯度来更新模型的参数，以最小化损失。

*   **SGD (Stochastic Gradient Descent):**
    *   $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$
    *   $\eta$ 是学习率。
    *   **PyTorch:** `torch.optim.SGD` (可附带 momentum, weight_decay)。
*   **Adam (Adaptive Moment Estimation):** 结合了 Momentum 和 RMSprop 的优点，是目前广泛使用的优化器。
    *   维护每个参数的一阶矩估计 (梯度的指数移动平均) 和二阶矩估计 (梯度平方的指数移动平均)。
    *   **PyTorch:** `torch.optim.Adam`, `torch.optim.AdamW` (Adam with decoupled weight decay)。
*   **其他常用优化器:** `Adagrad`, `RMSprop`, `Adadelta`。
*   **PyTorch 实现:** `torch.optim` 模块

    ```python
    model = nn.Linear(10, 2) # 示例模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay 即 L2 正则化

    # 训练循环中:
    # optimizer.zero_grad()   # 清除旧梯度
    # loss.backward()         # 反向传播计算梯度
    # optimizer.step()        # 更新参数
    ```

### 3. 反向传播 (Backpropagation)

PyTorch 的 `autograd` 引擎会自动计算梯度。当在 `Tensor` 上调用 `.backward()` 时，PyTorch 会根据计算图自动计算所有 `requires_grad=True` 的叶子节点的梯度，并累积到它们的 `.grad` 属性中。

### 4. 学习率调度 (Learning Rate Scheduling)

在训练过程中动态调整学习率，有助于模型更好地收敛和避免陷入局部最优。

*   **常用策略：**
    *   **StepLR:** 每隔几个 epoch 将学习率乘以一个因子。
    *   **ReduceLROnPlateau:** 当某个指标 (如验证集损失) 不再改善时，降低学习率。
    *   **CosineAnnealingLR:** 学习率随余弦函数周期性变化。
    *   **Warmup:** 训练初期使用较小的学习率，然后逐渐增加到预设值。
*   **PyTorch 实现:** `torch.optim.lr_scheduler`

    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # 每30个epoch，学习率乘以0.1

    # 训练循环中，在 optimizer.step() 之后调用:
    # scheduler.step()
    ```

## 四、正则化 (Regularization)

正则化用于防止模型过拟合，提高其在未见过数据上的泛化能力。

### 1. L1/L2 正则化 (Weight Decay)

通过在损失函数中添加参数的范数惩罚项。

*   **L2 正则化 (权重衰减):** $L_{total} = L_{original} + \frac{\lambda}{2} \sum_i w_i^2$
    *   倾向于使权重值变小，但不为零。
    *   在 PyTorch 优化器中通过 `weight_decay` 参数实现。
*   **L1 正则化:** $L_{total} = L_{original} + \lambda \sum_i |w_i|$
    *   倾向于产生稀疏权重（很多权重为零）。
    *   通常需要手动添加到损失中。

### 2. Dropout

在训练过程中，以一定的概率 $p$ 随机将一部分神经元的输出置为零，从而减少神经元之间的共同适应 (co-adaptation)。

*   **原理：** 类似于训练多个不同的“瘦”网络，并在测试时进行模型平均。
*   **PyTorch 实现:** `torch.nn.Dropout(p)`

    ```python
    dropout_layer = nn.Dropout(p=0.5)
    # 在模型的前向传播中应用
    # x = dropout_layer(x)
    # 注意: 在评估/测试阶段 (model.eval())，Dropout 会自动失效，所有神经元都会被使用，
    # 通常输出会乘以 (1-p) 来补偿训练时的丢弃（或者训练时将保留的激活值除以 (1-p)）。
    # PyTorch 的 nn.Dropout 在训练时是将保留的激活值除以 (1-p)，测试时直接使用。
    ```
    在 RNN/LSTM/GRU 层中，`dropout` 参数用于在除了最后一层之外的每个 RNN 层的输出上应用 dropout。

### 3. 早停 (Early Stopping)

在训练过程中监控验证集上的性能 (如损失或准确率)。当验证集性能在一定数量的 epoch 内不再提升时，提前停止训练，以防止过拟合。这通常需要手动实现。

### 4. 批量归一化 (Batch Normalization)

`torch.nn.BatchNorm1d` (对于序列数据，通常作用于特征维度)。虽然在 NLP 中不如 CV 中普遍，但有时用于词嵌入层或 RNN/Transformer 层的输出之后，以稳定训练和加速收敛。Transformer 中的 Layer Normalization (`torch.nn.LayerNorm`) 更为常见。

## 五、处理变长序列 (Handling Variable Length Sequences)

NLP 中的一个常见问题是输入序列长度不同。

*   **填充 (Padding):** 将同一批次内的所有序列填充到相同的最大长度。使用一个特殊的 `padding_idx`。
*   **`torch.nn.utils.rnn.pack_padded_sequence`:**
    *   在将填充后的序列输入 RNN/LSTM/GRU 之前，可以对其进行“打包”，使得 RNN 只处理实际的非填充部分，提高效率并确保结果正确。
    *   输入需要按序列长度降序排列（如果 `enforce_sorted=True`，默认为 `True`）。
*   **`torch.nn.utils.rnn.pad_packed_sequence`:**
    *   将打包后的 RNN 输出解包回填充的张量形式。

```python
# 假设 embedding_layer 和 rnn_layer 已经定义
# input_padded: [batch_size, max_seq_len, embedding_dim]
# seq_lengths: [batch_size] 存储每个序列的真实长度

# 为了使用 pack_padded_sequence，最好按长度排序（如果 enforce_sorted=True）
# sorted_lengths, sorted_idx = seq_lengths.sort(0, descending=True)
# sorted_input_padded = input_padded[sorted_idx]

packed_input = nn.utils.rnn.pack_padded_sequence(input_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
# 注意: seq_lengths 需要在CPU上

packed_output, (hidden, cell) = rnn_layer(packed_input) # 假设是LSTM

output_padded, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
# output_padded 现在是填充后的张量
# output_lengths 是原始的长度信息
```

## 六、PyTorch NLP 源码片段示例 (文本分类)
这是一个简化的文本分类模型，使用 Embedding + LSTM + Linear 层。
```python
import torch
import torch.nn as nn
import torch.optim as optim

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0, # LSTM自带的dropout只在多层时有效
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) # 额外的dropout层

    def forward(self, text, text_lengths):
        # text = [batch_size, seq_len]
        # text_lengths = [batch_size]

        embedded = self.dropout(self.embedding(text))
        # embedded = [batch_size, seq_len, embedding_dim]

        # 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        # cell = [num_layers * num_directions, batch_size, hidden_dim]

        # 解包输出 (如果需要中间所有时间步的输出)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 取最后时间步的隐藏状态 (对于双向LSTM，需要拼接最后的前向和后向隐藏状态)
        if self.lstm.bidirectional:
            # hidden 是 (num_layers * 2, batch, hidden_dim)
            # 取最后一层 (正向: -2, 反向: -1)
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # hidden 是 (num_layers, batch, hidden_dim)
            hidden = self.dropout(hidden[-1,:,:])
        # hidden = [batch_size, hidden_dim * num_directions]

        return self.fc(hidden)

# 示例参数
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2 # 二分类
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = 1 # 假设padding token的索引是1

model = TextClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

# 假设的输入
batch_size = 4
max_seq_len = 20
dummy_text = torch.randint(0, VOCAB_SIZE, (batch_size, max_seq_len))
dummy_text_lengths = torch.tensor([15, 18, 10, max_seq_len]) # 每个序列的真实长度

# 将PAD_IDX填充到短序列
for i in range(batch_size):
    if dummy_text_lengths[i] < max_seq_len:
        dummy_text[i, dummy_text_lengths[i]:] = PAD_IDX


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss() # 假设是分类任务

# 训练步骤 (伪代码)
# model.train()
# optimizer.zero_grad()
# predictions = model(dummy_text, dummy_text_lengths)
# dummy_labels = torch.tensor() # 假设的标签
# loss = criterion(predictions, dummy_labels)
# loss.backward()
# optimizer.step()

print(model)
# 可以通过 print(model.embedding.weight) 查看 Embedding 层的权重等。
# 要深入理解某一层 (如 LSTM) 的源码，可以直接在 PyTorch 官方 GitHub 仓库中搜索对应的类定义。
# e.g., pytorch/torch/nn/modules/rnn.py
```
