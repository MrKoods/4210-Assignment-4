#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: cnn.py
# SPECIFICATION: CNN to classify handwritten digits (32x32 images)
# FOR: CS 4210 - Assignment #4
# TIME SPENT: 
#-------------------------------------------------------------------------

# Importing Python libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):

        # Get label from filename (first character)
        label = int(filename[0])

        # Load image, convert to grayscale, resize
        img = Image.open(os.path.join(folder_path, filename)).convert('L').resize(image_size)

        X.append(np.array(img))
        y.append(label)

    return np.array(X), np.array(y)


# Paths
train_path = os.path.join("images", "train")
test_path = os.path.join("images", "test")

# Load data
X_train, Y_train = load_digit_images_from_folder(train_path)
X_test, Y_test = load_digit_images_from_folder(test_path)

# Normalize (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

# Build CNN model
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, Y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, Y_test)

# Print accuracy
print("Test accuracy:", acc)