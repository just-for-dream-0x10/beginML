# PyTorch NLP Core Notes

## Introduction

PyTorch, with its flexibility and dynamic graph mechanism, is widely used in both academia and industry for Natural Language Processing (NLP). These notes aim to summarize the core knowledge points موسيقى (mūsīqá - music) when developing NLP tasks using PyTorch.

## I. Text Representation

Computers cannot directly understand text, so text needs to be converted into numerical representations.

### 1. Word Embeddings

Word embeddings map discrete words to low-dimensional continuous vector spaces, such that semantically similar words are close in the vector space.

*   **Underlying Principle:** Based on the distributional hypothesis—a word's meaning is determined by its context. Word vectors are learned by studying the relationship between words and their contexts.
*   **Model Design:**
    *   **Word2Vec (Mikolov et al., 2013)**
        *   **CBOW (Continuous Bag-of-Words):** Predicts the center word based on context words.
            *   Objective function (simplified): Maximize the conditional probability of the center word given its context.
            *   $J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$
        *   **Skip-gram:** Predicts context words based on the center word.
            *   Objective function (simplified): Maximize the conditional probability of context words given the center word.
            *   $J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t)$
            *   Negative Sampling is commonly used to optimize computational efficiency.
    *   **GloVe (Pennington et al., 2014)**
        *   Principle: Utilizes global word co-occurrence statistics.
        *   Objective function: $J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$
            *   $w_i$: center word vector, $\tilde{w}_j$: context word vector, $X_{ij}$: co-occurrence count of word $i$ and word $j$, $f(X_{ij})$: weighting function.
    *   **FastText (Bojanowski et al., 2017)**
        *   Principle: Treats words as a collection of character n-grams. Vector representations are learned for each n-gram, and the word vector is the sum of its n-gram vectors. Handles Out-Of-Vocabulary (OOV) words better.
*   **PyTorch Implementation:**
    *   `torch.nn.Embedding(num_embeddings, embedding_dim)`
    *   `num_embeddings`: Size of the vocabulary.
    *   `embedding_dim`: Dimension of the word vectors.
    *   This module is essentially a lookup table, with a weight matrix $W \in \mathbb{R}^{\text{vocab_size} \times \text{embedding_dim}}$. Given a word's index, it outputs the corresponding word vector.
    *   The `padding_idx` parameter can specify the index of the padding token, whose vector will not be updated during training (usually a zero vector).
    *   Pre-trained word vectors (like Word2Vec, GloVe) can be loaded to initialize the weights of the `Embedding` layer.

    ```python
    import torch
    import torch.nn as nn

    # Assuming vocabulary size is 10000, embedding dimension is 300
    vocab_size = 10000
    embedding_dim = 300
    embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    # Assuming input is a sequence of word indices (batch_size=2, seq_len=5)
    input_indices = torch.tensor([,], dtype=torch.long)
    embedded_vectors = embedding_layer(input_indices)
    # embedded_vectors.shape will be (2, 5, 300)
    print(embedded_vectors.shape)
    ```

## II. Core Sequence Models

### 1. Recurrent Neural Networks (RNN)

RNNs are specialized for processing sequential data, capturing temporal dependencies through a recurrent structure.

*   **Underlying Principle:** The hidden state $h_t$ at the current time step is determined by the input $x_t$ at the current time step and the hidden state $h_{t-1}$ from the previous time step.
*   **Mathematical Formulas (Simple RNN):**
    *   $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
    *   $y_t = W_{hy}h_t + b_y$
    *   $W_{hh}, W_{xh}, W_{hy}$ are weight matrices, $b_h, b_y$ are bias terms, $\tanh$ is the activation function.
*   **Model Design:**
    *   Can stack multiple RNN layers (Stacked RNN) to learn more complex representations.
    *   Can be unidirectional or bidirectional. Bidirectional RNNs consider both past and future context.
*   **Issues:**
    *   **Vanishing Gradients:** In long sequences, gradients can exponentially decay during backpropagation, making it difficult for the model to learn long-term dependencies.
    *   **Exploding Gradients:** Gradients can exponentially grow, leading to unstable training. Gradient Clipping is a common mitigation technique.
*   **PyTorch Implementation:** `torch.nn.RNN`

    ```python
    # Input dimension 50, hidden layer dimension 100, 2-layer RNN
    input_size = 50
    hidden_size = 100
    num_layers = 2
    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    # batch_first=True means the input dimension order is (batch, seq, feature)

    # Assuming input (batch=3, seq_len=10, feature_dim=50)
    input_seq = torch.randn(3, 10, input_size)
    # Initial hidden state (num_layers * num_directions, batch, hidden_size)
    h0 = torch.randn(num_layers * 2, 3, hidden_size) # *2 for bidirectional
    output, hn = rnn(input_seq, h0)
    # output.shape: (3, 10, hidden_size * 2)
    # hn.shape: (num_layers * 2, 3, hidden_size)
    print(output.shape, hn.shape)
    ```

### 2. Long Short-Term Memory (LSTM)

LSTM is a special type of RNN that addresses the vanishing gradient problem by introducing gating mechanisms, thus better capturing long-term dependencies.

*   **Underlying Principle:** Introduces three gates (forget gate, input gate, output gate) and a cell state $C_t$ to control the flow and retention of information.
*   **Mathematical Formulas:**
    *   Forget Gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ (decides what information to discard from the cell state)
    *   Input Gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ (decides which new information to store in the cell state)
    *   Candidate Cell State: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
    *   Cell State Update: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ (discards some old information, adds some new information)
    *   Output Gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ (decides what part of the cell state to output)
    *   Hidden State: $h_t = o_t \odot \tanh(C_t)$
    *   $\sigma$ is the Sigmoid function, $\odot$ denotes element-wise product.
*   **Model Design:** Similar to RNN, can be stacked and bidirectional.
*   **PyTorch Implementation:** `torch.nn.LSTM` (parameters similar to `nn.RNN`, but returns `output, (hn, cn)`)

    ```python
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    # Initial hidden state and cell state
    h0 = torch.randn(num_layers * 2, 3, hidden_size)
    c0 = torch.randn(num_layers * 2, 3, hidden_size)
    output, (hn, cn) = lstm(input_seq, (h0, c0))
    print(output.shape, hn.shape, cn.shape)
    ```

### 3. Gated Recurrent Unit (GRU)

GRU is a variant of LSTM with a simpler structure, fewer parameters, and often higher computational efficiency, with performance comparable to LSTM.

*   **Underlying Principle:** Combines the forget and input gates of LSTM into an update gate, and merges the cell state and hidden state.
*   **Mathematical Formulas:**
    *   Reset Gate: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
    *   Update Gate: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
    *   Candidate Hidden State: $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$
    *   Hidden State: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
*   **PyTorch Implementation:** `torch.nn.GRU` (parameters similar to `nn.RNN`, returns `output, hn`)

    ```python
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    h0 = torch.randn(num_layers * 2, 3, hidden_size)
    output, hn = gru(input_seq, h0)
    print(output.shape, hn.shape)
    ```

### 4. Transformer (Vaswani et al., 2017 "Attention Is All You Need")

The Transformer model is entirely based on the self-attention mechanism, abandoning the recurrent structure of RNNs, thus achieving better parallelization and SOTA results on many NLP tasks.

*   **Underlying Principle:**
    *   **Self-Attention Mechanism:** Calculates the relevance (weights) of each word in a sequence to all other words, then computes a weighted sum to get the new representation for that word.
        *   **Scaled Dot-Product Attention:**
            $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
            *   $Q$ (Query), $K$ (Key), $V$ (Value) are matrices obtained by different linear transformations of the input sequence.
            *   $d_k$ is the dimension of the Key vector, used for scaling to prevent dot products from becoming too large and pushing softmax into regions with small gradients.
    *   **Multi-Head Attention:** Projects $Q, K, V$ into multiple different subspaces, performs attention calculations separately in each, and then concatenates the results followed by another linear transformation. This allows the model to attend to information from different positions in different representation subspaces.
        $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$
        where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
    *   **Positional Encoding:** Since the Transformer has no recurrent structure, it cannot capture sequence order information. Therefore, positional encodings need to be added to the input embeddings. Sine and cosine functions are commonly used:
        $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
        $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
        *   $pos$ is the position index, $i$ is the dimension index, $d_{model}$ is the embedding dimension.
*   **Model Design (Encoder-Decoder Architecture):**
    *   **Encoder:** Composed of N identical layers stacked, each layer containing a multi-head self-attention sub-layer and a position-wise feed-forward network sub-layer. Each sub-layer is followed by a residual connection and layer normalization.
    *   **Decoder:** Also composed of N identical layers stacked. Each layer, in addition to the two sub-layers of the Encoder, inserts an extra multi-head attention sub-layer that attends to the Encoder's output (Encoder-Decoder Attention). The self-attention layer in the Decoder needs to use masking to prevent the current position from attending to future positions (in sequence generation tasks).
*   **PyTorch Implementation:**
    *   `torch.nn.Transformer` (Complete Encoder-Decoder model)
    *   `torch.nn.TransformerEncoder`, `torch.nn.TransformerDecoder`
    *   `torch.nn.TransformerEncoderLayer`, `torch.nn.TransformerDecoderLayer`
    *   `torch.nn.MultiheadAttention`

    ```python
    # Example: Using TransformerEncoderLayer
    d_model = 512  # Model dimension (embedding_dim)
    nhead = 8      # Number of heads in multi-head attention
    num_encoder_layers = 6
    dim_feedforward = 2048 # Hidden layer dimension of the feed-forward network
    dropout = 0.1

    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    # Assuming input (batch=3, seq_len=10, feature_dim=d_model)
    src = torch.rand(3, 10, d_model)
    # Optional src_key_padding_mask to indicate which are padding tokens
    # src_key_padding_mask = torch.tensor([[False, False, False, True, True], ...], dtype=torch.bool)
    output = transformer_encoder(src) #, src_key_padding_mask=src_key_padding_mask)
    # output.shape: (3, 10, d_model)
    print(output.shape)

    # For direct use of MultiheadAttention
    multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
    # Assuming Q, K, V are all src
    attn_output, attn_output_weights = multihead_attn(src, src, src)
    # attn_output.shape: (3, 10, d_model)
    # attn_output_weights.shape: (3, 10, 10) (attention weights of each query to all keys)
    print(attn_output.shape, attn_output_weights.shape)
    ```

## III. Training Rules

### 1. Loss Functions

Loss functions measure the difference between model predictions and true labels.

*   **Cross-Entropy Loss:** Commonly used for classification tasks (e.g., text classification, per-timestep classification in sequence labeling).
    *   Formula (for a single sample, multi-class): $L = -\sum_{c=1}^{M} y_c \log(p_c)$
        *   $M$ is the number of classes, $y_c$ is an indicator variable (1 if the true class is $c$, 0 otherwise), $p_c$ is the model's predicted probability for class $c$.
    *   **PyTorch Implementation:**
        *   `torch.nn.CrossEntropyLoss`: Internally combines `LogSoftmax` and `NLLLoss`. Input is raw logits.
        *   `torch.nn.NLLLoss` (Negative Log Likelihood Loss): Input is log-probabilities.
        *   `torch.nn.BCELoss` (Binary Cross-Entropy Loss): For binary classification tasks.
        *   `torch.nn.BCEWithLogitsLoss`: Combines Sigmoid and BCELoss, numerically more stable.

    ```python
    criterion = nn.CrossEntropyLoss()
    # Assuming model output logits (batch_size=4, num_classes=3)
    outputs = torch.randn(4, 3, requires_grad=True)
    # True labels (batch_size=4)
    labels = torch.tensor(, dtype=torch.long)
    loss = criterion(outputs, labels)
    print(loss)
    ```
*   **Other Losses:**
    *   **Mean Squared Error Loss:** `torch.nn.MSELoss` (common for regression tasks).
    *   **CTC Loss (Connectionist Temporal Classification):** For sequence labeling tasks where alignment is variable (e.g., speech recognition). `torch.nn.CTCLoss`.

### 2. Optimizers

Optimizers update the model's parameters based on the gradients computed from the loss function to minimize the loss.

*   **SGD (Stochastic Gradient Descent):**
    *   $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$
    *   $\eta$ is the learning rate.
    *   **PyTorch:** `torch.optim.SGD` (can include momentum, weight_decay).
*   **Adam (Adaptive Moment Estimation):** Combines the advantages of Momentum and RMSprop, currently a widely used optimizer.
    *   Maintains first-moment estimates (exponential moving average of gradients) and second-moment estimates (exponential moving average of squared gradients) for each parameter.
    *   **PyTorch:** `torch.optim.Adam`, `torch.optim.AdamW` (Adam with decoupled weight decay).
*   **Other common optimizers:** `Adagrad`, `RMSprop`, `Adadelta`.
*   **PyTorch Implementation:** `torch.optim` module

    ```python
    model = nn.Linear(10, 2) # Example model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay is L2 regularization

    # In the training loop:
    # optimizer.zero_grad()   # Clear old gradients
    # loss.backward()         # Backpropagate to compute gradients
    # optimizer.step()        # Update parameters
    ```

### 3. Backpropagation

PyTorch's `autograd` engine automatically computes gradients. When `.backward()` is called on a `Tensor`, PyTorch automatically computes gradients for all leaf nodes with `requires_grad=True` based on the computation graph and accumulates them into their `.grad` attributes.

### 4. Learning Rate Scheduling

Dynamically adjusting the learning rate during training can help the model converge better and avoid getting stuck in local optima.

*   **Common Strategies:**
    *   **StepLR:** Multiplies the learning rate by a factor every few epochs.
    *   **ReduceLROnPlateau:** Reduces the learning rate when a metric (e.g., validation loss) stops improving.
    *   **CosineAnnealingLR:** Learning rate varies periodically according to a cosine function.
    *   **Warmup:** Uses a smaller learning rate in the initial phase of training, then gradually increases it to the preset value.
*   **PyTorch Implementation:** `torch.optim.lr_scheduler`

    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # Every 30 epochs, lr is multiplied by 0.1

    # In the training loop, after optimizer.step():
    # scheduler.step()
    ```

## IV. Regularization

Regularization is used to prevent model overfitting and improve its generalization ability on unseen data.

### 1. L1/L2 Regularization (Weight Decay)

Adds a penalty term of the parameter norms to the loss function.

*   **L2 Regularization (Weight Decay):** $L_{total} = L_{original} + \frac{\lambda}{2} \sum_i w_i^2$
    *   Tends to make weight values smaller, but not necessarily zero.
    *   Implemented in PyTorch optimizers via the `weight_decay` parameter.
*   **L1 Regularization:** $L_{total} = L_{original} + \lambda \sum_i |w_i|$
    *   Tends to produce sparse weights (many weights become zero).
    *   Usually needs to be manually added to the loss.

### 2. Dropout

During training, randomly sets the output of some neurons to zero with a probability $p$, thereby reducing co-adaptation between neurons.

*   **Principle:** Similar to training multiple different "thinned" networks and performing model averaging during testing.
*   **PyTorch Implementation:** `torch.nn.Dropout(p)`

    ```python
    dropout_layer = nn.Dropout(p=0.5)
    # Applied in the model's forward pass
    # x = dropout_layer(x)
    # Note: During evaluation/testing (model.eval()), Dropout is automatically disabled, all neurons are used.
    # Typically, the output is multiplied by (1-p) to compensate for the dropout during training
    # (or, during training, the activations of retained units are scaled by 1/(1-p)).
    # PyTorch's nn.Dropout scales by 1/(1-p) during training and does nothing during evaluation.
    ```
    In RNN/LSTM/GRU layers, the `dropout` parameter applies dropout to the outputs of each RNN layer except the last one.

### 3. Early Stopping

Monitors performance on a validation set (e.g., loss or accuracy) during training. If the validation performance does not improve for a certain number of epochs, training is stopped early to prevent overfitting. This usually requires manual implementation.

### 4. Batch Normalization

`torch.nn.BatchNorm1d` (for sequence data, typically applied to the feature dimension). While not as common in NLP as in CV, it is sometimes used after embedding layers or RNN/Transformer layer outputs to stabilize training and accelerate convergence. Layer Normalization (`torch.nn.LayerNorm`) is more common in Transformers.

## V. Handling Variable Length Sequences

A common issue in NLP is that input sequences have different lengths.

*   **Padding:** Pad all sequences in the same batch to the same maximum length. Use a special `padding_idx`.
*   **`torch.nn.utils.rnn.pack_padded_sequence`:**
    *   Before feeding a padded sequence into an RNN/LSTM/GRU, it can be "packed" so that the RNN only processes the actual non-padded parts, improving efficiency and ensuring correct results.
    *   Input needs to be sorted by sequence length in descending order (if `enforce_sorted=True`, which is the default).
*   **`torch.nn.utils.rnn.pad_packed_sequence`:**
    *   Unpacks the output of a packed RNN back into a padded tensor form.

```python
# Assume embedding_layer and rnn_layer are defined
# input_padded: [batch_size, max_seq_len, embedding_dim]
# seq_lengths: [batch_size] stores the true length of each sequence

# To use pack_padded_sequence, it's best to sort by length (if enforce_sorted=True)
# sorted_lengths, sorted_idx = seq_lengths.sort(0, descending=True)
# sorted_input_padded = input_padded[sorted_idx]

# Note: seq_lengths needs to be on CPU
packed_input = nn.utils.rnn.pack_padded_sequence(input_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)


packed_output, (hidden, cell) = rnn_layer(packed_input) # Assuming it's an LSTM

output_padded, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
# output_padded is now the padded tensor
# output_lengths contains the original length information
```

## VI. PyTorch NLP Source Code Snippet Example (Text Classification)
This is a simplified text classification model using Embedding + LSTM + Linear layers.

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
                            dropout=dropout if n_layers > 1 else 0, # LSTM's built-in dropout only works for n_layers > 1
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) # Additional dropout layer

    def forward(self, text, text_lengths):
        # text = [batch_size, seq_len]
        # text_lengths = [batch_size]

        embedded = self.dropout(self.embedding(text))
        # embedded = [batch_size, seq_len, embedding_dim]

        # Pack sequence
        # text_lengths must be on CPU
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        # cell = [num_layers * num_directions, batch_size, hidden_dim]

        # Unpack output (if all intermediate timesteps' outputs are needed)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Take the hidden state of the last time step (for bidirectional LSTM, concatenate the last forward and backward hidden states)
        if self.lstm.bidirectional:
            # hidden is (num_layers * 2, batch, hidden_dim)
            # Take the last layer (forward: -2, backward: -1)
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # hidden is (num_layers, batch, hidden_dim)
            hidden = self.dropout(hidden[-1,:,:])
        # hidden = [batch_size, hidden_dim * num_directions]

        return self.fc(hidden)

# Example parameters
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2 # Binary classification
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = 1 # Assuming padding token index is 1

model = TextClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

# Dummy input
batch_size = 4
max_seq_len = 20
dummy_text = torch.randint(0, VOCAB_SIZE, (batch_size, max_seq_len))
dummy_text_lengths = torch.tensor([15, 18, 10, max_seq_len]) # True length of each sequence

# Fill PAD_IDX into shorter sequences
for i in range(batch_size):
    if dummy_text_lengths[i] < max_seq_len:
        dummy_text[i, dummy_text_lengths[i]:] = PAD_IDX


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss() # Assuming a classification task

# Training step (pseudo-code)
# model.train()
# optimizer.zero_grad()
# predictions = model(dummy_text, dummy_text_lengths)
# dummy_labels = torch.tensor() # Dummy labels
# loss = criterion(predictions, dummy_labels)
# loss.backward()
# optimizer.step()

print(model)
# You can check the weights of the Embedding layer with print(model.embedding.weight), etc.
# To understand the source code of a specific layer (e.g., LSTM) in depth,
# you can search for the corresponding class definition in the official PyTorch GitHub repository.
# e.g., pytorch/torch/nn/modules/rnn.py
```
