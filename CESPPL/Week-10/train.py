"""
Train the CESPPL image classifier.

Examples
--------
Run a two-epoch smoke test:

    python train.py --epochs 2

Train with MobileNetV2:

    python train.py --backbone mobilenetv2 --epochs 10
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data import (
    get_class_names,
    get_class_weights,
    load_split,
)

from src.model import (
    SUPPORTED_BACKBONES,
    build_model,
)


tf.keras.utils.set_random_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CESPPL image classifier."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnetb0",
        choices=SUPPORTED_BACKBONES,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--class_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cesppl_training",
    )

    return parser.parse_args()


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CESPPL CLASSIFIER TRAINING")
    print("=" * 60)

    class_names = get_class_names()

    print("\nClass order:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    print("\nLoading training dataset...")

    train_dataset = load_split(
        split_name="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    print("Loading validation dataset...")

    val_dataset = load_split(
        split_name="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    print("Datasets loaded successfully.")

    class_weight_dict = None

    if args.class_weights:
        class_weight_dict = get_class_weights()

        print("\nClass weights:")
        for index, weight in class_weight_dict.items():
            print(
                f"{index}: {class_names[index]} = {weight:.4f}"
            )

    print("\nBuilding model...")

    model = build_model(
        num_classes=len(class_names),
        backbone=args.backbone,
        img_size=args.img_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )

    model.summary()

    best_weights_path = output_dir / "best.weights.h5"
    final_weights_path = output_dir / "final.weights.h5"
    history_path = output_dir / "history.json"
    config_path = output_dir / "training_config.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    model.save_weights(str(final_weights_path))

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    save_json(
        history_dict,
        history_path,
    )

    training_config = {
        "random_seed": 42,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "class_weights_enabled": args.class_weights,
        "class_names": class_names,
        "class_weights": class_weight_dict,
    }

    save_json(
        training_config,
        config_path,
    )

    print("\nTraining completed successfully.")
    print(f"Outputs saved in: {output_dir.resolve()}")

    print("\nCreated files:")
    print(best_weights_path)
    print(final_weights_path)
    print(history_path)
    print(config_path)


if __name__ == "__main__":
    main()