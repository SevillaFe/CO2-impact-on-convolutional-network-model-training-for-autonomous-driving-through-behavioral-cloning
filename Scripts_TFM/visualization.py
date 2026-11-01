from __future__ import annotations

import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_steering(y_train, y_valid):
    """Plot steering angle histograms for train and validation splits."""
    max_degree = 25
    degree_per_steering = 10
    n_classes = max_degree * degree_per_steering
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
    ax0, ax1 = axes.flatten()

    ax0.hist(y_train, bins=n_classes, histtype="bar", color="blue", rwidth=0.6, label="train")
    ax0.set_title("Training set")
    ax0.set_xlabel("Steering Angle")
    ax0.set_ylabel("Total Image")

    ax1.hist(y_valid, bins=n_classes, histtype="bar", color="red", rwidth=0.6, label="valid")
    ax1.set_title("Validation set")
    ax1.set_xlabel("Steering Angle")
    ax1.set_ylabel("Total Image")

    fig.tight_layout()
    plt.show()


def show_steering_2(y_train, y_valid):
    """Plot train and validation steering distributions with Plotly."""
    max_degree = 25
    degree_per_steering = 10
    n_classes = max_degree * degree_per_steering
    counts_t, bins = np.histogram(y_train, bins=n_classes)
    counts_v, bins = np.histogram(y_valid, bins=n_classes)
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Train Samples", "Validation samples"])
    fig.append_trace(go.Bar(x=bins, y=counts_t, name="Train"), row=1, col=1)
    fig.append_trace(go.Bar(x=bins, y=counts_v, name="Validation"), row=1, col=2)
    fig.show()


def show_example(func, X_train=None, steerings=None):
    """Visualize a preprocessing or augmentation function on a sample image."""
    if X_train is None or steerings is None:
        raise ValueError("X_train and steerings are required to visualize examples.")

    random_index = random.randint(0, min(1000, len(X_train) - 1))
    image = X_train[random_index]
    steering_angle = steerings[random_index]
    original_image = mpimg.imread(image)

    if func.__name__ == "random_flip":
        preprocessed_image, flipped_steering_angle = func(original_image, steering_angle)
    else:
        preprocessed_image = func(original_image)

    print(f"Original shape: {original_image.shape}, Preprocessed shape: {preprocessed_image.shape}")
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()

    if func.__name__ == "random_flip":
        axs[0].imshow(original_image)
        axs[0].set_title(f"Original Image - Steering Angle: {steering_angle}")
        axs[1].imshow(preprocessed_image)
        axs[1].set_title(f"Flipped Image - Steering Angle: {flipped_steering_angle}")
    else:
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[1].imshow(preprocessed_image)
        axs[1].set_title("Preprocessed Image")


__all__ = ["show_steering", "show_steering_2", "show_example"]
