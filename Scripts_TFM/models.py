from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def NvidiaModel() -> Sequential:
    """PilotNet-inspired model for steering regression."""
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
    return model


def VGG() -> Sequential:
    """VGG-style convolutional network for steering prediction."""
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(66, 200, 3), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(20))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
    return model


__all__ = ["NvidiaModel", "VGG"]
