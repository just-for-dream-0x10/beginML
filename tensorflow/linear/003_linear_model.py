import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Normalization
import seaborn as sns


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]
data = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

data = data.drop("origin", axis=1)
print(data.isna().sum())
data = data.dropna()


train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop("mpg")
test_labels = test_features.pop("mpg")

# use Normalization layer of Keras to normalize data
data_normalizer = Normalization(axis=1)
data_normalizer.adapt(np.array(train_features))


model = K.Sequential(
    [
        data_normalizer,
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation=None),
    ]
)
model.summary()

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(train_features, train_labels, epochs=100, validation_split=0.2)

# Make predictions
y_pred = model.predict(test_features).flatten()

# Plotting section
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot training & validation loss values
axes[0, 0].plot(history.history["loss"], label="Train Loss")
axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
axes[0, 0].set_title("Model Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Error [MPG]")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Scatter plot of true vs predicted values
axes[0, 1].scatter(test_labels, y_pred)
axes[0, 1].set_xlabel("True Values [MPG]")
axes[0, 1].set_ylabel("Predictions [MPG]")
axes[0, 1].set_title("True vs Predicted Values")
lims = [0, 50]
axes[0, 1].set_xlim(lims)
axes[0, 1].set_ylim(lims)
axes[0, 1].plot(lims, lims)

# Histogram of prediction errors
error = y_pred - test_labels
axes[1, 0].hist(error, bins=30)
axes[1, 0].set_xlabel("Prediction Error [MPG]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Prediction Error Distribution")

# Pairplot (as a separate figure since it's complex)
sns.pairplot(
    train_dataset[
        [
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
        ]
    ],
    diag_kind="kde",
)

OUTPUT_DIR = "./notes/images"

plt.savefig(f"{OUTPUT_DIR}/linear_model.png")

plt.tight_layout()
plt.show()
