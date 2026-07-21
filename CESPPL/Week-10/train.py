import argparse
import json
from datetime import datetime
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


SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the CESPPL image classifier."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnetb0",
        choices=SUPPORTED_BACKBONES,
        help="Backbone architecture.",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout value.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--class_weights",
        action="store_true",
        help="Enable balanced class weights.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cesppl_training",
        help="Directory used to save training outputs.",
    )

    parser.add_argument(
        "--sanity_overfit",
        action="store_true",
        help=(
            "Overfit one batch of 32 images for 100 epochs "
            "with augmentation disabled."
        ),
    )

    return parser.parse_args()


def make_json_serializable(history_dictionary):
    clean_history = {}

    for key, values in history_dictionary.items():
        clean_history[key] = [
            float(value)
            for value in values
        ]

    return clean_history


def create_sanity_dataset(
    img_size,
    batch_size=32,
):
    """
    Load exactly 32 training images.

    The same batch is used repeatedly so the model should
    eventually memorize it.
    """

    full_train_dataset = load_split(
        split_name="train",
        img_size=img_size,
        batch_size=batch_size,
    )

    sanity_dataset = (
        full_train_dataset
        .unbatch()
        .take(32)
        .batch(32)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return sanity_dataset


def run_sanity_overfit(
    args,
    class_names,
    output_directory,
):
    print("\n" + "=" * 60)
    print("SANITY OVERFIT TEST")
    print("=" * 60)

    print("\nLoading exactly 32 training images...")

    sanity_dataset = create_sanity_dataset(
        img_size=args.img_size,
        batch_size=32,
    )

    print("Augmentation: disabled")
    print("Backbone: trainable")
    print("Epochs: 100")
    print("Target: training loss should approach zero\n")

    model = build_model(
        backbone_name=args.backbone,
        num_classes=len(class_names),
        img_size=args.img_size,
        dropout=0.0,
        learning_rate=args.learning_rate,
        use_augmentation=False,
        backbone_trainable=True,
    )

    model.summary()

    history = model.fit(
        sanity_dataset,
        epochs=100,
        verbose=1,
    )

    final_loss = float(
        history.history["loss"][-1]
    )

    final_accuracy = float(
        history.history["accuracy"][-1]
    )

    print("\n" + "=" * 60)
    print("SANITY TEST RESULT")
    print("=" * 60)

    print(f"Final training loss: {final_loss:.8f}")
    print(f"Final training accuracy: {final_accuracy:.4f}")

    if final_loss < 0.05 and final_accuracy >= 0.95:
        sanity_status = "PASSED"

        print("\nSanity check PASSED.")
        print(
            "The model successfully memorized the small batch."
        )

    else:
        sanity_status = "NEEDS REVIEW"

        print("\nSanity check did not fully pass.")
        print(
            "Check image loading, labels, preprocessing, "
            "loss function and model configuration."
        )

    sanity_result = {
        "status": sanity_status,
        "backbone": args.backbone,
        "number_of_images": 32,
        "epochs": 100,
        "augmentation": False,
        "backbone_trainable": True,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "completed_at": datetime.now().isoformat(),
    }

    sanity_json_path = (
        output_directory / "sanity_overfit_result.json"
    )

    with open(
        sanity_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sanity_result,
            file,
            indent=4,
        )

    history_path = (
        output_directory / "sanity_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(history.history),
            file,
            indent=4,
        )

    model.save_weights(
        output_directory / "sanity_final.weights.h5"
    )

    print(
        f"\nSanity result saved to:\n{sanity_json_path}"
    )


def run_normal_training(
    args,
    class_names,
    output_directory,
):
    print("\nLoading training dataset...")

    train_dataset = load_split(
        split_name="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    print("Loading validation dataset...")

    validation_dataset = load_split(
        split_name="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    print("Datasets loaded successfully.")

    class_weight_dictionary = None

    if args.class_weights:
        class_weight_dictionary = get_class_weights()

        print("\nClass weights:")

        for index, class_name in enumerate(class_names):
            print(
                f"{index}: {class_name} = "
                f"{class_weight_dictionary[index]:.4f}"
            )

    else:
        print("\nClass weights: disabled")

    print("\nBuilding model...")

    model = build_model(
        backbone_name=args.backbone,
        num_classes=len(class_names),
        img_size=args.img_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        use_augmentation=True,
        backbone_trainable=False,
    )

    model.summary()

    best_weights_path = (
        output_directory / "best.weights.h5"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weight_dictionary,
        callbacks=callbacks,
        verbose=1,
    )

    final_weights_path = (
        output_directory / "final.weights.h5"
    )

    model.save_weights(final_weights_path)

    history_path = output_directory / "history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(history.history),
            file,
            indent=4,
        )

    training_config = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "requested_epochs": args.epochs,
        "completed_epochs": len(
            history.history["loss"]
        ),
        "batch_size": args.batch_size,
        "class_weights": args.class_weights,
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "completed_at": datetime.now().isoformat(),
    }

    config_path = (
        output_directory / "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_config,
            file,
            indent=4,
        )

    print("\nTraining completed successfully.")

    print(
        f"Outputs saved in: "
        f"{output_directory.resolve()}"
    )

    print("\nCreated files:")

    print(best_weights_path)
    print(final_weights_path)
    print(history_path)
    print(config_path)


def main():
    args = parse_arguments()

    tf.keras.utils.set_random_seed(SEED)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("CESPPL CLASSIFIER TRAINING")
    print("=" * 60)

    class_names = get_class_names()

    print("\nClass order:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    if args.sanity_overfit:
        run_sanity_overfit(
            args=args,
            class_names=class_names,
            output_directory=output_directory,
        )

    else:
        run_normal_training(
            args=args,
            class_names=class_names,
            output_directory=output_directory,
        )


if __name__ == "__main__":
    main()