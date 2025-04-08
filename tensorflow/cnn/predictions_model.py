import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input,
    decode_predictions,
)
from PIL import Image
import numpy as np
import os

test_images_dir = "./test_images"
img_names = [
    os.path.join(test_images_dir, f)
    for f in os.listdir(test_images_dir)
    if f.endswith((".png", ".jpg", ".jpeg"))
]

# Modern approach to load and resize images
imgs = []
for img_name in img_names:
    # Open image with PIL
    img = Image.open(img_name)
    # Resize to 224x224 for EfficientNet
    img = img.resize((224, 224))
    # Convert to numpy array and preprocess properly
    img_array = np.array(img).astype("float32")
    img_array = preprocess_input(img_array)
    imgs.append(img_array)

imgs = np.array(imgs)

# Use EfficientNetB0 instead of ResNet50 or VGG16
model = EfficientNetB0(weights="imagenet", include_top=True)
print(model.summary())
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# predict
predictions = model.predict(imgs)
decoded_predictions = decode_predictions(predictions, top=3)

# Print image names alongside predictions
for i, img_name in enumerate(img_names):
    print(f"\nImage: {os.path.basename(img_name)}")
    print("Top predictions:")
    for j, (imagenet_id, label, score) in enumerate(decoded_predictions[i]):
        print(f"  {j+1}: {label} ({score:.2f})")
