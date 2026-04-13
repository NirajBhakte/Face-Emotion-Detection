"""
train.py — Optimized for Face Emotion Detection (7 classes, FER-2013)
Fixes: class imbalance, weak architecture, no callbacks, underfitting
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# Config
# ==============================
TRAIN_DIR   = "data/train"
TEST_DIR    = "data/test"
MODEL_PATH  = "models/emotion_model.h5"
IMG_SIZE    = 48
BATCH_SIZE  = 64        # larger batch → more stable gradients
EPOCHS      = 60        # more epochs; EarlyStopping will terminate wisely
LR          = 1e-3

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==============================
# Data Augmentation
# ==============================
# FIX: stronger augmentation helps with only 7 classes and imbalanced data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\n✅ Class Index Mapping (SAVE THIS — must match live_detection.py):")
print(train_data.class_indices)
# Expected: {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

# ==============================
# Class Weights (fixes imbalance)
# ==============================
# FIX: FER-2013 has ~500 Disgust vs ~8000 Happy samples → huge imbalance
# class_weight tells the model to penalize errors on rare classes more
labels = train_data.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_array))
print("\n✅ Class Weights (higher = rarer class):")
for idx, label in enumerate(EMOTION_LABELS):
    print(f"  {label}: {class_weights[idx]:.3f}")

# ==============================
# Improved CNN Architecture
# ==============================
# FIX: Added BatchNormalization, deeper network, more Dense units
# BatchNorm stabilizes training and acts as a regularizer
def build_model(num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # Block 4 — extra depth for 7 classes
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # Classifier Head
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(train_data.num_classes)
model.summary()

# ==============================
# Compile
# ==============================
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Callbacks
# ==============================
# FIX: No callbacks was causing blind training; these three are essential
callbacks = [
    # Save only the best model (by val_accuracy)
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if val_accuracy doesn't improve for 10 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Halve LR if val_loss stagnates for 5 epochs
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ==============================
# Train
# ==============================
print("\n🚀 Starting training...\n")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights   # FIX: handles class imbalance
)

print(f"\n✅ Best model saved to: {MODEL_PATH}")

# ==============================
# Evaluation
# ==============================
test_data.reset()
loss, acc = model.evaluate(test_data, verbose=0)
print(f"\n📊 Final Test Accuracy: {acc*100:.2f}%")
print(f"📊 Final Test Loss:     {loss:.4f}")

# ==============================
# Plot Accuracy & Loss
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Acc',  color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val Acc',    color='orange')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/training_plots.png", dpi=150)
plt.show()

print("\n✅ Training complete! Plots saved to outputs/training_plots.png")