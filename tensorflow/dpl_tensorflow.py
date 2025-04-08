import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values; we --> reshape it to 60000 x 784.
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# Normalize inputs to be within in [0, 1].
X_train /= 255
X_test /= 255
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
# One-hot representation of the labels.
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model.
model = tf.keras.models.Sequential(
    [
        keras.layers.Input(shape=(RESHAPED,)),
        keras.layers.Dense(
            N_HIDDEN,
            name="dense_layer",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
        keras.layers.Dropout(0.3),  # Add dropout to prevent overfitting
        keras.layers.Dense(
            N_HIDDEN,
            name="dense_layer_2",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
        keras.layers.Dropout(0.3),  # Add dropout to prevent overfitting
        keras.layers.Dense(
            N_HIDDEN,
            name="dense_layer_3",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NB_CLASSES, name="dense_layer_4", activation="softmax"),
    ]
)

print(f"summary model: {model.summary()}\n")

# model.compile(optimizer="RMSProp", loss="categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Implement early stopping to optimize model overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=early_stopping,
)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)


print("TensorFlow version:", tf.__version__)
print("mps", tf.config.list_physical_devices("mps"))

# show the training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()
