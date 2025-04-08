import numpy as np
import os
import shutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from help_function import download_and_read_commond


# create model
class SentimentAnalysisModel(tf.keras.Model):
    def __init__(self, vocab_size, max_seqlen, **kwargs):
        super(SentimentAnalysisModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, max_seqlen)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(max_seqlen, return_sequences=False)  # Changed to False
        )
        self.dense = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def build(self, input_shape):
        # Properly implement build method
        self.embedding.build(input_shape)
        embedding_output_shape = (input_shape[0], input_shape[1], input_shape[1])
        self.bilstm.build(embedding_output_shape)
        # Output shape is now (batch_size, 2*max_seqlen) since return_sequences=False
        bilstm_output_shape = (input_shape[0], 2 * input_shape[1])
        self.dense.build(bilstm_output_shape)
        dense_output_shape = (input_shape[0], 64)
        self.out.build(dense_output_shape)
        self.built = True

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = self.bilstm(x)  # Now returns (batch_size, 2*max_seqlen)
        x = self.dense(x)  # Now returns (batch_size, 64)
        x = self.out(x)  # Now returns (batch_size, 1)
        return x


if __name__ == "__main__":

    # Download and read the labeled sentences
    labeled_sentences = download_and_read_commond(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        + "00331/sentiment%20labelled%20sentences.zip"
    )

    sentences = [s for (s, l) in labeled_sentences]
    labels = [int(l) for (s, l) in labeled_sentences]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index)
    print("vocabulary size: {:d}".format(vocab_size))
    word2idx = tokenizer.word_index
    idx2word = {v: k for (k, v) in word2idx.items()}

    seq_lengths = np.array([len(s.split()) for s in sentences])
    print([(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])

    max_seqlen = 64
    # create dataset for model
    sentences_as_ints = tokenizer.texts_to_sequences(sentences)
    sentences_as_ints = tf.keras.preprocessing.sequence.pad_sequences(
        sentences_as_ints, maxlen=max_seqlen
    )
    labels_as_ints = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, labels_as_ints))

    dataset = dataset.shuffle(10000)
    test_size = len(sentences) // 3
    val_size = (len(sentences) - test_size) // 10
    test_dataset = dataset.take(test_size)
    val_dataset = dataset.skip(test_size).take(val_size)
    train_dataset = dataset.skip(test_size + val_size)
    batch_size = 64
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    model = SentimentAnalysisModel(vocab_size + 1, max_seqlen)
    model.build(input_shape=(batch_size, max_seqlen))
    """
    Model: "sentiment_analysis_model"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ bidirectional (Bidirectional)        │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ ?                           │     0 (unbuilt) │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_1 (Dense)                      │ ?                           │     0 (unbuilt) │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 0 (0.00 B)
    Trainable params: 0 (0.00 B)
    Non-trainable params: 0 (0.00 B)
    """
    print(model.summary())
    # compile
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    data_dir = "./data"
    # Make sure the directory exists
    os.makedirs(data_dir, exist_ok=True)
    logs_dir = os.path.join("./logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Fix the file extension to match the save_weights_only=True requirement
    best_model_file = os.path.join(data_dir, "best_model.weights.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_file, save_weights_only=True, save_best_only=True
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
    num_epochs = 10
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint, tensorboard],
    )

    best_model = SentimentAnalysisModel(vocab_size + 1, max_seqlen)
    best_model.build(input_shape=(batch_size, max_seqlen))
    best_model.load_weights(best_model_file)
    best_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    test_loss, test_acc = best_model.evaluate(test_dataset)
    print("test loss: {:.3f}, test accuracy: {:.3f}".format(test_loss, test_acc))

    labels, predictions = [], []
    idx2word[0] = "PAD"
    is_first_batch = True
    for test_batch in test_dataset:
        inputs_b, labels_b = test_batch
        pred_batch = best_model.predict(inputs_b)
        predictions.extend([(1 if p > 0.5 else 0) for p in pred_batch])
        labels.extend([l for l in labels_b])
        if is_first_batch:
            # print first batch of label, prediction, and sentence
            for rid in range(inputs_b.shape[0]):
                words = [idx2word[idx] for idx in inputs_b[rid].numpy()]
                words = [w for w in words if w != "PAD"]
                sentence = " ".join(words)
                print(
                    "{:d}\t{:d}\t{:s}".format(labels[rid], predictions[rid], sentence)
                )
            is_first_batch = False
    print("accuracy score: {:.3f}".format(accuracy_score(labels, predictions)))
    print("confusion matrix")
    print(confusion_matrix(labels, predictions))
