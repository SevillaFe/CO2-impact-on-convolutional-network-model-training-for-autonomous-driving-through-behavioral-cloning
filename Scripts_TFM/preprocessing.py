from __future__ import annotations

import random
from typing import Tuple

import cv2
import matplotlib.image as mpimg
import numpy as np
from imgaug import augmenters as iaa


def zoom(image: np.ndarray) -> np.ndarray:
    """Apply a random zoom transformation."""
    zoom_transform = iaa.Affine(scale=(1, 1.5))
    return zoom_transform.augment_image(image)


def random_flip(image: np.ndarray, steering_angle: float) -> Tuple[np.ndarray, float]:
    """Flip the image horizontally and invert the steering angle."""
    image = cv2.flip(image, 1)
    return image, -steering_angle


def pan(image: np.ndarray) -> np.ndarray:
    """Apply small translations to the image."""
    pan_transform = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan_transform.augment_image(image)


def random_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly change the brightness of the image."""
    brightness = iaa.Multiply((0.2, 1.6))
    return brightness.augment_image(image)


def image_brighten(image: np.ndarray) -> np.ndarray:
    """Adjust brightness using HSV value channel."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    default_bias = 0.25
    brightness_factor = default_bias + np.random.uniform()
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_factor
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def random_shadow(image: np.ndarray) -> np.ndarray:
    """Add a synthetic shadow to simulate different lighting conditions."""
    image_height, image_width, _ = image.shape
    x1, y1 = image_width * np.random.rand(), 0
    x2, y2 = image_width * np.random.rand(), image_height
    xm, ym = np.mgrid[0:image_height, 0:image_width]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_augment(image_path: str, steering_angle: float) -> Tuple[np.ndarray, float]:
    """Randomly apply a set of image augmentations and adjust steering angle."""
    image = mpimg.imread(image_path)

    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    if np.random.rand() < 0.5:
        image = image_brighten(image)
    if np.random.rand() < 0.5:
        image = random_shadow(image)
    if np.random.rand() < 0.5:
        image, steering_angle = random_flip(image, steering_angle)

    return image, steering_angle


def img_preprocess(img: np.ndarray) -> np.ndarray:
    """Crop, resize, normalize and convert the image to the expected CNN input format."""
    img = img[40:140, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66), cv2.INTER_AREA)
    img = (img - 128.0) / 128.0
    return img


__all__ = [
    "zoom",
    "random_flip",
    "pan",
    "random_brightness",
    "image_brighten",
    "random_shadow",
    "random_augment",
    "img_preprocess",
]
