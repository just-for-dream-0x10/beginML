import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# network config
#  learning rate
initial_learning_rate = 0.001
lr_schedule = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)

EPOCHS = 10
BATCH_ZIZE = 64
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
VALIDATION_SPLIT = 0.2

IMG_ROWS, IMG_COLS = 28, 28  # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NUM_CLASSES = 10  # number of outputs = number of digits


def build_model():
    model = models.Sequential()

    # 1 block
    model.add(
        layers.Conv2D(
            32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation="relu"
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # drop 20% nodes
    model.add(layers.Dropout(0.2))

    # 2nd block
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    # 3d block
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    # dense 3D ====> 1D features
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
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
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

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

    # data augmentation
    # 算是对数据进行处理，让模型更加的健壮=======> 数据增强
    datagen = ImageDataGenerator(
        # 图像旋转角度 + - 30度
        rotation_range=30,
        # 水平平移的范围是宽度的 + - 20%
        width_shift_range=0.2,
        # 垂直平移的范围是高度的 + - 20%
        height_shift_range=0.2,
        # 随机对图像进行翻转
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    model = build_model()
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

    # Uncomment the training code
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_ZIZE),
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Create models directory if it doesn't exist
    import os

    os.makedirs("./models", exist_ok=True)

    # Fix the weights filename format
    model.save_weights("./models/model.weights.h5")

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
