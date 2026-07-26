import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data import (
    get_class_names,
    load_split,
)

from src.model import (
    SUPPORTED_BACKBONES,
    build_model,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained "
            "CESPPL classifier."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the "
            ".weights.h5 checkpoint."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=[
            "train",
            "val",
            "test",
        ],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnetb0",
        choices=SUPPORTED_BACKBONES,
        help=(
            "Backbone used "
            "during training."
        ),
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help=(
            "Image size used "
            "during training."
        ),
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help=(
            "Dropout used "
            "during training."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help=(
            "Directory for "
            "evaluation outputs."
        ),
    )

    return parser.parse_args()


def collect_predictions(
    model,
    dataset,
):
    true_labels = []
    predicted_labels = []
    all_probabilities = []
    all_image_paths = []

    for batch in dataset:
        if len(batch) == 3:
            (
                image_batch,
                label_batch,
                path_batch,
            ) = batch

            batch_paths = [
                path.decode("utf-8")
                for path
                in path_batch.numpy()
            ]

            all_image_paths.extend(
                batch_paths
            )

        elif len(batch) == 2:
            (
                image_batch,
                label_batch,
            ) = batch

            all_image_paths.extend(
                [""] * len(label_batch)
            )

        else:
            raise ValueError(
                "Dataset must return either "
                "(images, labels) or "
                "(images, labels, paths)."
            )

        probabilities = model.predict(
            image_batch,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        true_labels.extend(
            label_batch.numpy().tolist()
        )

        predicted_labels.extend(
            predictions.tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

    return (
        np.array(true_labels),
        np.array(predicted_labels),
        np.array(all_probabilities),
        np.array(all_image_paths),
    )


def save_confusion_matrix(
    matrix,
    class_names,
    output_path,
):
    figure_size = 12

    plt.figure(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "CESPPL Confusion Matrix"
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )

    tick_positions = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_positions,
        class_names,
        rotation=90,
    )

    plt.yticks(
        tick_positions,
        class_names,
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            plt.text(
                column_index,
                row_index,
                str(value),
                horizontalalignment="center",
                color=text_color,
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_per_class_metrics(
    report,
    class_names,
    output_path,
):
    fieldnames = [
        "class",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for class_name in class_names:
            class_metrics = report[
                class_name
            ]

            writer.writerow(
                {
                    "class": class_name,
                    "precision": (
                        class_metrics[
                            "precision"
                        ]
                    ),
                    "recall": (
                        class_metrics[
                            "recall"
                        ]
                    ),
                    "f1_score": (
                        class_metrics[
                            "f1-score"
                        ]
                    ),
                    "support": (
                        class_metrics[
                            "support"
                        ]
                    ),
                }
            )


def main():
    args = parse_arguments()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    output_directory = Path(
        args.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names = get_class_names()

    print("=" * 60)
    print("CESPPL MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nCheckpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Split: "
        f"{args.split}"
    )

    print(
        f"Backbone: "
        f"{args.backbone}"
    )

    print(
        "\nLoading evaluation dataset..."
    )

    dataset = load_split(
        split_name=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        include_paths=True,
    )

    print("Building model...")

    model = build_model(
        backbone_name=args.backbone,
        num_classes=len(class_names),
        img_size=args.img_size,
        dropout=args.dropout,
        learning_rate=1e-3,
        use_augmentation=False,
        backbone_trainable=False,
    )

    print("Loading checkpoint...")

    model.load_weights(
        checkpoint_path
    )

    print("Running predictions...")

    (
        true_labels,
        predicted_labels,
        probabilities,
        image_paths,
    ) = collect_predictions(
        model=model,
        dataset=dataset,
    )

    if len(image_paths) != len(
        true_labels
    ):
        raise ValueError(
            "Number of image paths does "
            "not match the number of "
            "predictions."
        )

    label_indices = np.arange(
        len(class_names)
    )

    overall_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nOverall accuracy: "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Macro F1-score: "
        f"{macro_f1:.4f}"
    )

    print("\nPer-class metrics:")

    for class_name in class_names:
        metrics = report[
            class_name
        ]

        print(
            f"\n{class_name}"
            f"\n  Precision: "
            f"{metrics['precision']:.4f}"
            f"\n  Recall:    "
            f"{metrics['recall']:.4f}"
            f"\n  F1-score:  "
            f"{metrics['f1-score']:.4f}"
            f"\n  Support:   "
            f"{int(metrics['support'])}"
        )

    summary = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "split": args.split,
        "backbone": args.backbone,
        "overall_accuracy": float(
            overall_accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "number_of_samples": int(
            len(true_labels)
        ),
        "per_class_metrics": {
            class_name: {
                "precision": float(
                    report[class_name][
                        "precision"
                    ]
                ),
                "recall": float(
                    report[class_name][
                        "recall"
                    ]
                ),
                "f1_score": float(
                    report[class_name][
                        "f1-score"
                    ]
                ),
                "support": int(
                    report[class_name][
                        "support"
                    ]
                ),
            }
            for class_name
            in class_names
        },
    }

    metrics_json_path = (
        output_directory
        / "evaluation_metrics.json"
    )

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    per_class_csv_path = (
        output_directory
        / "per_class_metrics.csv"
    )

    save_per_class_metrics(
        report=report,
        class_names=class_names,
        output_path=per_class_csv_path,
    )

    confusion_matrix_path = (
        output_directory
        / "confusion_matrix.png"
    )

    save_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=confusion_matrix_path,
    )

    predictions_path = (
        output_directory
        / "predictions.csv"
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "image_path",
                "true_index",
                "true_class",
                "predicted_index",
                "predicted_class",
                "top1_class",
                "top1_confidence",
                "top2_class",
                "top2_confidence",
                "top3_class",
                "top3_confidence",
                "correct",
            ]
        )

        for (
            image_path,
            true_index,
            predicted_index,
            probability_values,
        ) in zip(
            image_paths,
            true_labels,
            predicted_labels,
            probabilities,
        ):
            top_three_indices = np.argsort(
                probability_values
            )[-3:][::-1]

            top_three_confidences = (
                probability_values[
                    top_three_indices
                ]
                * 100
            )

            writer.writerow(
                [
                    str(image_path),
                    int(true_index),
                    class_names[
                        int(true_index)
                    ],
                    int(predicted_index),
                    class_names[
                        int(predicted_index)
                    ],
                    class_names[
                        int(
                            top_three_indices[0]
                        )
                    ],
                    round(
                        float(
                            top_three_confidences[
                                0
                            ]
                        ),
                        2,
                    ),
                    class_names[
                        int(
                            top_three_indices[1]
                        )
                    ],
                    round(
                        float(
                            top_three_confidences[
                                1
                            ]
                        ),
                        2,
                    ),
                    class_names[
                        int(
                            top_three_indices[2]
                        )
                    ],
                    round(
                        float(
                            top_three_confidences[
                                2
                            ]
                        ),
                        2,
                    ),
                    bool(
                        true_index
                        == predicted_index
                    ),
                ]
            )

    print(
        "\nEvaluation completed successfully."
    )

    print("\nCreated files:")
    print(metrics_json_path)
    print(per_class_csv_path)
    print(confusion_matrix_path)
    print(predictions_path)


if __name__ == "__main__":
    main()