import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt

# network config
#  learning rate
initial_learning_rate = 0.001
lr_schedule = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)

EPOCHS = 20
BATCH_ZIZE = 123
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
VALIDATION_SPLIT = 0.90

IMG_ROWS, IMG_COLS = 28, 28  # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10  # number of outputs = number of digits


def build_model(input_shape, classes):
    # function API not Sequential API
    inputs = tf.keras.Input(shape=input_shape)

    # CONV => RELU => POOL
    x = layers.Conv2D(
        32, (3, 3), activation="relu", kernel_regularizer=K.regularizers.l2(0.001)
    )(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    drop = layers.Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = layers.Conv2D(
        64, (3, 3), activation="relu", kernel_regularizer=K.regularizers.l2(0.001)
    )(drop)
    x = layers.MaxPooling2D((2, 2))(x)
    drops = layers.Dropout(0.25)(x)
    # convert to 1D (flatten) full connection layer
    x = layers.Flatten()(drops)
    x = layers.Dense(
        300, activation="relu", kernel_regularizer=K.regularizers.l2(0.001)
    )(x)
    drops = layers.Dropout(0.25)(x)
    # softmax classifier
    outputs = layers.Dense(classes, activation="softmax")(drops)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model


def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    # normalize
    X_train, X_test = X_train / 255.0, X_test / 255.0
    # cast
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":

    if tf.config.list_physical_devices("GPU"):
        print("GPU is available")
    else:
        print("GPU is not available, checking for MPS...")
        if tf.config.list_physical_devices("MPS"):
            print("MPS is available")

            tf.config.experimental.set_visible_devices(
                tf.config.list_physical_devices("MPS")[0], "MPS"
            )

            tf.config.experimental.set_memory_growth(
                tf.config.list_physical_devices("MPS")[0], True
            )
        else:
            print("No GPU or MPS available, using CPU")

    (X_train, y_train), (X_test, y_test) = load_data()
    model = build_model(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(
        loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"]
    )

    print(model.summary())
    early_stopping = early_stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    callbacks = [
        # Write TensorBoard logs to './logs' directory
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs",
            histogram_freq=1,  # How often to write histogram visualizations
            write_graph=True,  # Whether to write model graph as a protobuf file
            write_images=True,  # Whether to write model weights as image files
            embeddings_freq=1,  # How often to write embedding visualizations
            update_freq="epoch",  # "batch" or "epoch" or integer
            profile_batch=2,  # Profile the computation every Nth batch
        ),
        early_stopping,
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_ZIZE,
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )

    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    print("\nTest score:", score[0])
    print("Test accuracy:", score[1])

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
    # plt.savefig(f"{OUTPUT_DIR}/{MODEL_NAME}.png")
    plt.show()
    plt.close()
