import nltk
import numpy as np
import os
import shutil
import tensorflow.keras as k
import tensorflow as tf
from help_function import download_and_read_for_mtm


# Define masked accuracy function to ignore padding tokens
def masked_accuracy():
    def accuracy(y_true, y_pred):
        # Find positions where there is a real token (not padding)
        mask = tf.math.not_equal(tf.math.reduce_sum(y_true, axis=-1), 0)
        # Get the class with the highest probability
        y_pred_class = tf.math.argmax(y_pred, axis=-1)
        y_true_class = tf.math.argmax(y_true, axis=-1)
        # Compare predictions with true values only where mask is True
        matches = tf.math.equal(y_pred_class, y_true_class)
        matches = tf.cast(matches, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        # Compute accuracy using the mask to ignore padding
        return tf.math.reduce_sum(matches * mask) / tf.math.reduce_sum(mask)

    return accuracy


class POSTaggerModel(tf.keras.Model):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        embedding_dim,
        max_seqlen,
        rnn_output_dim,
        **kwargs,
    ):
        super(POSTaggerModel, self).__init__(**kwargs)
        self.embed = tf.keras.layers.Embedding(
            source_vocab_size, embedding_dim, input_length=max_seqlen
        )
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(rnn_output_dim, return_sequences=True)
        )
        # Change target_vocab_size to target_vocab_size + 1 to match the target shape
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(target_vocab_size + 1)
        )
        self.activation = tf.keras.layers.Activation("softmax")

    def call(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x = self.rnn(x)
        x = self.dense(x)
        x = self.activation(x)
        return x


def tokenize_and_build_vocab(texts, vocab_size=None, lower=True):
    if vocab_size is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=lower)
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size + 1, oov_token="UNK", lower=lower
        )
    tokenizer.fit_on_texts(texts)

    if vocab_size is not None:
        # additional workaround, see issue 8092
        # https://github.com/keras-team/keras/issues/8092
        tokenizer.word_index = {
            e: i for e, i in tokenizer.word_index.items() if i <= vocab_size + 1
        }
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, tokenizer


# Download the required NLTK dataset if not already downloaded
try:
    nltk.data.find("corpora/treebank")
except LookupError:
    print("Downloading NLTK treebank dataset...")
    nltk.download("treebank")

if __name__ == "__main__":
    # Check if treebank is properly loaded
    print("Checking treebank availability...")
    try:
        # Verify treebank is accessible by getting a sample
        sample = nltk.corpus.treebank.tagged_sents()
        print(f"Treebank loaded successfully. Found {len(sample)} sentences.")
    except Exception as e:
        print(f"Error accessing treebank: {e}")

    # Get the data
    sents, poss = download_and_read_for_mtm("./datasets")

    # Check if data was loaded
    if len(sents) == 0:
        print("Warning: No sentences were loaded. Checking datasets directory...")
        os.makedirs("./datasets", exist_ok=True)

        # Try to directly extract from NLTK corpus
        print("Attempting to extract directly from NLTK corpus...")
        sentences = nltk.corpus.treebank.tagged_sents()

        # Process and save sentences
        sent_filename = os.path.join("./datasets", "treebank-sents.txt")
        poss_filename = os.path.join("./datasets", "treebank-poss.txt")

        with open(sent_filename, "w") as fsents, open(poss_filename, "w") as fposs:
            for sent in sentences:
                if sent:  # Make sure the sentence is not empty
                    fsents.write(" ".join([w for w, p in sent]) + "\n")
                    fposs.write(" ".join([p for w, p in sent]) + "\n")

        # Read the newly created files
        sents, poss = [], []
        with open(sent_filename, "r") as fsent:
            for line in fsent:
                sents.append(line.strip())
        with open(poss_filename, "r") as fposs:
            for line in fposs:
                poss.append(line.strip())

    assert len(sents) == len(poss)
    print("# of records: {:d}".format(len(sents)))

    # Continue with the rest of your code
    word2idx_s, idx2word_s, tokenizer_s = tokenize_and_build_vocab(
        sents, vocab_size=9000
    )
    word2idx_t, idx2word_t, tokenizer_t = tokenize_and_build_vocab(
        poss, vocab_size=38, lower=False
    )
    source_vocab_size = len(word2idx_s)
    target_vocab_size = len(word2idx_t)
    print(
        "vocab sizes (source): {:d}, (target): {:d}".format(
            source_vocab_size, target_vocab_size
        )
    )

    sequence_lengths = np.array([len(s.split()) for s in sents])
    print([(p, np.percentile(sequence_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])

    max_seqlen = 271
    sents_as_ints = tokenizer_s.texts_to_sequences(sents)
    sents_as_ints = tf.keras.preprocessing.sequence.pad_sequences(
        sents_as_ints, maxlen=max_seqlen, padding="post"
    )
    # convert POS tags to sequence of (categorical) integers
    poss_as_ints = tokenizer_t.texts_to_sequences(poss)
    poss_as_ints = tf.keras.preprocessing.sequence.pad_sequences(
        poss_as_ints, maxlen=max_seqlen, padding="post"
    )
    poss_as_catints = []

    for p in poss_as_ints:
        poss_as_catints.append(
            tf.keras.utils.to_categorical(
                p,
                num_classes=target_vocab_size + 1,
                # Removed the dtype parameter that was causing the error
            )
        )
    poss_as_catints = tf.keras.preprocessing.sequence.pad_sequences(
        poss_as_catints, maxlen=max_seqlen
    )

    dataset = tf.data.Dataset.from_tensor_slices((sents_as_ints, poss_as_catints))
    idx2word_s[0], idx2word_t[0] = "PAD", "PAD"

    # split into training, validation, and test datasets
    dataset = dataset.shuffle(10000)
    test_size = len(sents) // 3
    val_size = (len(sents) - test_size) // 10
    test_dataset = dataset.take(test_size)
    val_dataset = dataset.skip(test_size).take(val_size)
    train_dataset = dataset.skip(test_size + val_size)
    # create batches
    batch_size = 128
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    embedding_dim = 128
    rnn_output_dim = 256
    model = POSTaggerModel(
        source_vocab_size, target_vocab_size, embedding_dim, max_seqlen, rnn_output_dim
    )
    model.build(input_shape=(batch_size, max_seqlen))
    """
    Model: "pos_tagger_model"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ spatial_dropout1d (SpatialDropout1D) │ ?                           │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ bidirectional (Bidirectional)        │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ time_distributed (TimeDistributed)   │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ activation (Activation)              │ ?                           │               0 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 0 (0.00 B)
    Trainable params: 0 (0.00 B)
    Non-trainable params: 0 (0.00 B)
    """
    print(model.summary())

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", masked_accuracy()],
    )

    # After the model.compile line, add:

    # Define data directory for saving model checkpoints
    data_dir = "./datasets"  # Use the constant from help_function.py
    os.makedirs(data_dir, exist_ok=True)

    # Define logs directory for TensorBoard
    logs_dir = os.path.join(data_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Setup model checkpoint callback
    best_model_file = os.path.join(data_dir, "best_model.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_file, save_best_only=True, monitor="val_accuracy", mode="max"
    )

    # Setup TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

    # Train the model
    num_epochs = 10
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint, tensorboard],
    )
