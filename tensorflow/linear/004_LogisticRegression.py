import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Flatten


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel(
        "Pred {} Conf: {:2.0f}% True ({})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# input 28 x 28 images =====> flatten to 784 one dimension vector

mnist = K.datasets.mnist

((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()

# e normalize the images by dividing each image by 255.0
# original pixel values are between 0 and 255 / 255 =  0===> 1
train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)
test_data = test_data / np.float32(255)
test_labels = test_labels.astype(np.int32)

#  learning rate
initial_learning_rate = 0.001
lr_schedule = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)


# input ---> x ---> hidden ---> output
inputs = K.layers.Input(shape=(28, 28))
x = Flatten()(inputs)
hidden = Dense(30, activation="relu", kernel_regularizer=K.regularizers.l2(0.001))(x)
dropout = K.layers.Dropout(0.2)(hidden)
outputs = Dense(10, activation="sigmoid")(dropout)

model = K.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    optimizer=K.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# early_stopping
# monitor = val_loss patience = 5 means if the validation loss does not improve for 5 epochs, the training will be stopped
# restore_best_weights = True means the weights of the model will be set to the weights of the epoch with the best validation loss
early_stopping = K.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


# verbose = 0 means silent mode
# verbose = 1 means progress bar
# verbose = 2 means one line per epoch
# validation_split = 0.2 means 20% of the training data will be used as validation data
# callbacks = [early_stopping] means early stopping will be used

history = model.fit(
    train_data,
    train_labels,
    epochs=50,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping],
)

model.save("./models/mnist_model.h5")

# show loss curve
# plt.plot(history.history["loss"], label="loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()


predictions = model.predict(test_data)
i = 56
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_data)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
