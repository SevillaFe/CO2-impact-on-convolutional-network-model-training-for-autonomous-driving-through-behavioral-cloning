from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from data_utils import batch_generator, loadMakeMoreCreateDF, remove_sample, trainValSample
from models import NvidiaModel, VGG


def resolve_default_data_dir() -> str:
    """Return the most likely default dataset location for this project."""
    candidates = [
        os.path.join(os.getcwd(), "TrainImagesTFM_1_2"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "dataset"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def build_model(model_name: str):
    """Return the selected architecture."""
    models: Dict[str, object] = {
        "vgg": VGG,
        "nvidia": NvidiaModel,
    }
    if model_name.lower() not in models:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {sorted(models.keys())}")
    return models[model_name.lower()]()


def prepare_training_data(data_dir: str, balance_samples: bool = True):
    """Create the training and validation datasets from the simulator data folder."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    csv_path = os.path.join(data_dir, "driving_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing driving_log.csv in: {data_dir}")

    df = loadMakeMoreCreateDF(data_dir)
    if balance_samples:
        df = remove_sample(df)

    _, _, X_train, X_valid, y_train, y_valid = trainValSample(data_dir, df)
    return X_train, X_valid, y_train, y_valid


def train_model(
    data_dir: str = None,
    model_name: str = "vgg",
    epochs: int = 20,
    batch_size: int = 16,
    output_dir: str = None,
) -> None:
    """Train the steering regression model for the given dataset."""
    if data_dir is None:
        data_dir = resolve_default_data_dir()
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "trained_models")

    X_train, X_valid, y_train, y_valid = prepare_training_data(data_dir)
    model = build_model(model_name)

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "seq_logs")
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    training_size = len(X_train)
    validation_size = len(X_valid)
    steps_per_epoch = int(np.ceil(1.0 * training_size / batch_size))
    validation_steps = int(np.ceil(1.0 * validation_size / batch_size))

    train_gen = batch_generator(X_train, y_train, batch_size, True)
    valid_gen = batch_generator(X_valid, y_valid, batch_size, False)

    print(f"Training with dataset: {data_dir}")
    print(f"Model: {model_name}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        callbacks=[tensorboard_callback],
    )

    model_path = os.path.join(output_dir, f"model_{model_name.lower()}.h5")
    model.save(model_path)

    print(f"Model saved to: {model_path}")
    print(f"Training loss: {history.history['loss'][-1]}")
    print(f"Validation loss: {history.history['val_loss'][-1]}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train a steering-prediction CNN for autonomous driving behavior cloning.")
    parser.add_argument("--data-dir", type=str, default=resolve_default_data_dir(), help="Path to the simulator dataset folder containing driving_log.csv and IMG/. Default: auto-detects common project folders.")
    parser.add_argument("--model", type=str, default="vgg", choices=["vgg", "nvidia"], help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used during training.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.getcwd(), "trained_models"), help="Directory where the trained model and logs will be saved.")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
