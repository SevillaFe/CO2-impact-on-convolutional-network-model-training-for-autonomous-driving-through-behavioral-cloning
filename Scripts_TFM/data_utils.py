from __future__ import annotations

import os
import random
from typing import List, Tuple

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from preprocessing import img_preprocess, random_augment


def createDataFrame(data_path: str) -> pd.DataFrame:
    """Create a driving dataframe from a CSV file."""
    data_frame = pd.read_csv(data_path)
    data_frame.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    return data_frame


def createTrainingDataPathsCLR(df: pd.DataFrame, prefix_path: str) -> Tuple[List[List[object]], List[List[object]], List[List[object]]]:
    """Group image paths by steering direction using the three camera inputs."""
    center_turns: List[List[object]] = []
    left_turns: List[List[object]] = []
    right_turns: List[List[object]] = []
    abs_path_to_img = os.path.abspath(prefix_path)

    for _, row in df.iterrows():
        center_image_cam = os.path.join(abs_path_to_img, row["center"].strip())
        left_image_cam = os.path.join(abs_path_to_img, row["left"].strip())
        right_image_cam = os.path.join(abs_path_to_img, row["right"].strip())
        steering_angle = row["steering"]

        if steering_angle > 0.125:
            right_turns.append([center_image_cam, left_image_cam, right_image_cam, steering_angle])
        elif steering_angle < -0.125:
            left_turns.append([center_image_cam, left_image_cam, right_image_cam, steering_angle])
        else:
            center_turns.append([center_image_cam, left_image_cam, right_image_cam, steering_angle])

    return center_turns, left_turns, right_turns


def makeMore(data_array: List[List[object]], amount: int) -> List[List[object]]:
    """Duplicate samples to balance the dataset."""
    for i in range(len(data_array)):
        for _ in range(amount):
            data_array.append([data_array[i][0], data_array[i][1], data_array[i][2], data_array[i][3]])
    return data_array


def pathleaf(path: str) -> str:
    """Return only the filename from a full path."""
    _, tail = os.path.split(path)
    return tail


def loadMakeMoreCreateDF(data_dir: str, center: int = 0, left: int = 1, right: int = 1, num_bins: int = 25) -> pd.DataFrame:
    """Create and balance the final dataframe from the simulator data folder."""
    csv_path = os.path.join(data_dir, "driving_log.csv")
    data_pd_udacity = createDataFrame(csv_path)
    columns = ["center", "left", "right", "steering"]

    data_pd_udacity["center"] = data_pd_udacity["center"].apply(pathleaf)
    data_pd_udacity["left"] = data_pd_udacity["left"].apply(pathleaf)
    data_pd_udacity["right"] = data_pd_udacity["right"].apply(pathleaf)

    center_turns, left_turns, right_turns = createTrainingDataPathsCLR(data_pd_udacity, os.path.join(data_dir, "IMG"))
    center_turns = makeMore(center_turns, center)
    left_turns = makeMore(left_turns, left)
    right_turns = makeMore(right_turns, right)

    df_center = shuffle(pd.DataFrame(center_turns, columns=columns))
    df_left = shuffle(pd.DataFrame(left_turns, columns=columns))
    df_right = shuffle(pd.DataFrame(right_turns, columns=columns))

    frames_to_concat = [df_center, df_left, df_right]
    return pd.concat(frames_to_concat, axis=0, ignore_index=False)


def remove_sample(df: pd.DataFrame, num_bins: int = 25, samples_per_bin: int = 1100) -> pd.DataFrame:
    """Trim overrepresented steering bins to reduce bias toward straight driving."""
    df = df.reset_index(drop=True)
    hist, bins = np.histogram(df["steering"], num_bins)
    remove_list = []

    for j in range(num_bins):
        list_ = []
        for i in range(len(df["steering"])):
            if bins[j] <= df["steering"][i] <= bins[j + 1]:
                list_.append(i)
        shuffled = shuffle(list_)
        remove_list.extend(shuffled[samples_per_bin:])

    df = df.drop(df.index[remove_list]).reset_index(drop=True)
    return df


def load_img_steering(data_dir: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build arrays of image paths and steering labels."""
    image_paths = []
    steering = []

    for _, indexed_data in df.iterrows():
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_paths.append(os.path.join(data_dir, center.strip()))
        steering.append(float(indexed_data[3]))

        image_paths.append(os.path.join(data_dir, left.strip()))
        steering.append(float(indexed_data[3]) + (0.25 if abs(float(indexed_data[3])) > 1 else 0.18))

        image_paths.append(os.path.join(data_dir, right.strip()))
        steering.append(float(indexed_data[3]) - (0.25 if abs(float(indexed_data[3])) > 1 else 0.18))

    return np.asarray(image_paths), np.asarray(steering)


def trainValSample(data_dir: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create training and validation splits."""
    image_paths, steerings = load_img_steering(os.path.join(data_dir, "IMG"), df)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=42)
    return image_paths, steerings, X_train, X_valid, y_train, y_valid


def normal_load(image_paths: np.ndarray, steering_angles: np.ndarray, batch_size: int, is_training: bool):
    """Load and preprocess a batch of images while optionally applying augmentation."""
    batch_img = []
    batch_steering = []

    for _ in range(batch_size):
        random_index = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[random_index]
        steering = float(steering_angles[random_index])

        if is_training and np.random.randint(6) == 0:
            batch_img.append(img_preprocess(mpimg.imread(image_path)))
            batch_steering.append(steering)

        image, steering = random_augment(image_path, steering)
        batch_img.append(img_preprocess(image))
        batch_steering.append(steering)

    return np.asarray(batch_img), np.asarray(batch_steering)


def batch_generator(image_paths: np.ndarray, steering_angles: np.ndarray, batch_size: int, is_training: bool):
    """Yield batches lazily to reduce memory consumption during training."""
    while True:
        batch_img = []
        batch_steering = []

        for _ in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            steering = float(steering_angles[random_index])

            if is_training:
                image, steering = random_augment(image_path, steering)
            else:
                image = mpimg.imread(image_path)

            batch_img.append(img_preprocess(image))
            batch_steering.append(steering)

        yield np.asarray(batch_img), np.asarray(batch_steering)


def loadDataNoGenerator(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray):
    """Load all data in RAM with preprocessing applied."""
    X_train_normal, y_train_normal = normal_load(X_train, y_train, len(X_train), True)
    X_valid_normal, y_valid_normal = normal_load(X_valid, y_valid, len(X_valid), False)
    return X_train_normal, y_train_normal, X_valid_normal, y_valid_normal


__all__ = [
    "createDataFrame",
    "createTrainingDataPathsCLR",
    "makeMore",
    "pathleaf",
    "loadMakeMoreCreateDF",
    "remove_sample",
    "load_img_steering",
    "trainValSample",
    "normal_load",
    "batch_generator",
    "loadDataNoGenerator",
]
