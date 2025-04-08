from help_function import (
    download_and_read,
    split_train_labels,
    generate_text,
    CHECKPOINT_DIR,
)
import os
from tensorflow import keras as K
import tensorflow as tf
import numpy as np
import shutil

EPOCHS = 50


class GRNmodels(K.Model):
    def __init__(
        self, vocab_size, num_timesteps, embedding_dim, batch_size=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = K.layers.Embedding(vocab_size, embedding_dim)
        self.gru = K.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            return_sequences=True,
            stateful=True,
            # Remove the incorrect batch_input_shape parameter
        )
        self.dense = K.layers.Dense(vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def build(self, input_shape):
        # This method will be called when the model is first used
        self.embedding.build(input_shape)
        # Build GRU with the correct shape after embedding
        embedding_output_shape = (input_shape[0], input_shape[1], self.embedding_dim)
        self.gru.build(embedding_output_shape)
        # Build dense layer
        gru_output_shape = (input_shape[0], input_shape[1], self.num_timesteps)
        self.dense.build(gru_output_shape)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.gru(x)
        x = self.dense(x)
        return x

    def reset_states(self):
        self.gru.reset_states()


def loss(labels, predictions):
    loss_fn = K.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn(labels, predictions)


if __name__ == "__main__":
    texts = download_and_read(
        [
            "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
            "https://www.gutenberg.org/files/12/12-0.txt",
        ]
    )
    # create vocabulary
    vocab = sorted(set(" ".join(texts)))
    print("vocab size: {:d}".format(len(vocab)))
    # create mapping from vocab chars to ints
    char2idx = {u: i for i, u in enumerate(vocab)}
    # create mapping from ints to vocab chars
    idx2char = {i: c for c, i in char2idx.items()}

    # convert text to tensors
    texts_as_int = np.array([char2idx[c] for c in texts])
    data = tf.data.Dataset.from_tensor_slices(texts_as_int)

    # number of characters to show before asking for prediction
    # sequences: [None, 100]
    seq_length = 100
    sequences = data.batch(seq_length + 1, drop_remainder=True)
    sequences = sequences.map(split_train_labels)
    # setup trian   batches: [None, 64, 100]
    batch_size = 64
    steps_per_epoch = len(texts) // seq_length // batch_size
    dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)

    #  setup model && build
    vocab_size = len(vocab)
    embedding_dim = 256

    # For training, use the batch size you defined earlier
    model = GRNmodels(
        vocab_size=vocab_size,
        num_timesteps=seq_length,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )

    # No need to call build explicitly, the model will be built when first called

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=loss,
    )
    """
    Model: "gr_nmodels"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ gru (GRU)                            │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ ?                           │     0 (unbuilt) │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 0 (0.00 B)
    Trainable params: 0 (0.00 B)
    Non-trainable params: 0 (0.00 B)
    """
    # print(model.summary())
    for i in range(EPOCHS // 10):
        model.fit(
            dataset.repeat(),
            epochs=10,
            steps_per_epoch=steps_per_epoch,
        )
        # Fix the checkpoint file path to use the correct extension
        checkpoint_file = os.path.join(
            CHECKPOINT_DIR, "model_epoch_{:d}.weights.h5".format(i + 1)
        )

        # Make sure the checkpoint directory exists
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        model.save_weights(checkpoint_file)

        # For text generation, use batch_size=1
        grn_model = GRNmodels(
            vocab_size=vocab_size,
            num_timesteps=seq_length,
            embedding_dim=embedding_dim,
            batch_size=1,  # Important for text generation
        )

        # Build the model with the correct input shape before loading weights
        # The input shape should match what we'll use for text generation
        dummy_input = tf.zeros((1, seq_length), dtype=tf.int32)
        _ = grn_model(dummy_input)  # This will trigger the build

        # Now load the weights after the model is built
        grn_model.load_weights(checkpoint_file)

        print("after epoch: {:d}".format((i + 1) * 10))
        print(generate_text(grn_model, "Alice ", char2idx, idx2char))
        print("---")
