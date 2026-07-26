from __future__ import annotations

from typing import Callable

import tensorflow as tf


keras = tf.keras
layers = tf.keras.layers


SUPPORTED_BACKBONES = {
    "efficientnetb0",
    "mobilenetv2",
    "resnet50v2",
}


def get_backbone(
    backbone_name: str,
    image_size: int,
) -> tuple[keras.Model, Callable]:
    """
    Create an ImageNet-pretrained backbone and return
    its corresponding preprocessing function.
    """

    backbone_name = backbone_name.lower().strip()
    input_shape = (image_size, image_size, 3)

    if backbone_name == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

        preprocess_input = (
            keras.applications.efficientnet.preprocess_input
        )

    elif backbone_name == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

        preprocess_input = (
            keras.applications.mobilenet_v2.preprocess_input
        )

    elif backbone_name == "resnet50v2":
        backbone = keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

        preprocess_input = (
            keras.applications.resnet_v2.preprocess_input
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from: {sorted(SUPPORTED_BACKBONES)}"
        )

    return backbone, preprocess_input


def build_augmentation(
    mode: str = "standard",
) -> keras.Sequential:
    """
    Create the data-augmentation pipeline.

    Available modes
    ---------------
    standard:
        Original augmentation used in the baseline runs.

    strong_lighting:
        Stronger brightness and contrast augmentation intended
        to improve performance on dark and unevenly lit images.
    """

    mode = mode.lower().strip()

    if mode == "standard":
        return keras.Sequential(
            [
                layers.RandomFlip(
                    mode="horizontal",
                    name="random_flip",
                ),
                layers.RandomRotation(
                    factor=0.08,
                    name="random_rotation",
                ),
                layers.RandomZoom(
                    height_factor=0.10,
                    width_factor=0.10,
                    name="random_zoom",
                ),
                layers.RandomContrast(
                    factor=0.10,
                    name="random_contrast",
                ),
            ],
            name="data_augmentation",
        )

    if mode == "strong_lighting":
        return keras.Sequential(
            [
                layers.RandomFlip(
                    mode="horizontal",
                    name="random_flip",
                ),
                layers.RandomRotation(
                    factor=0.12,
                    name="random_rotation",
                ),
                layers.RandomZoom(
                    height_factor=0.15,
                    width_factor=0.15,
                    name="random_zoom",
                ),
                layers.RandomContrast(
                    factor=0.20,
                    name="random_contrast",
                ),
                layers.RandomBrightness(
                    factor=0.20,
                    value_range=(0.0, 255.0),
                    name="random_brightness",
                ),
            ],
            name="data_augmentation",
        )

    raise ValueError(
        f"Unknown augmentation mode: {mode}. "
        "Choose either 'standard' or 'strong_lighting'."
    )


def build_model(
    num_classes: int,
    backbone_name: str = "efficientnetb0",
    image_size: int = 224,
    dropout_rate: float = 0.3,
    use_augmentation: bool = True,
    augmentation_mode: str = "standard",
    img_size: int | None = None,
    dropout: float | None = None,
    learning_rate: float = 1e-3,
    backbone_trainable: bool = False,
) -> keras.Model:
    """
    Build and compile a transfer-learning image classifier.

    The function supports both naming styles used in the project:

    - image_size or img_size
    - dropout_rate or dropout
    """

    backbone_name = backbone_name.lower().strip()

    # Support argument names used by train.py and evaluate.py.
    if img_size is not None:
        image_size = img_size

    if dropout is not None:
        dropout_rate = dropout

    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from: {sorted(SUPPORTED_BACKBONES)}"
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1."
        )

    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than 0."
        )

    if not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must be between 0 and 1."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="image",
    )

    x = inputs

    if use_augmentation:
        augmentation = build_augmentation(
            mode=augmentation_mode,
        )

        x = augmentation(x)

    backbone, preprocess_input = get_backbone(
        backbone_name=backbone_name,
        image_size=image_size,
    )

    backbone.trainable = backbone_trainable

    x = layers.Lambda(
        preprocess_input,
        name="preprocessing",
    )(x)

    # BatchNormalization statistics remain frozen while
    # the backbone is used during fine-tuning.
    x = backbone(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{backbone_name}_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def find_backbone(
    model: keras.Model,
) -> keras.Model:
    """
    Locate the pretrained application backbone inside the model.
    """

    for layer in model.layers:
        if not isinstance(layer, keras.Model):
            continue

        layer_name = layer.name.lower()

        if layer_name in SUPPORTED_BACKBONES:
            return layer

        # Additional robust checks in case Keras changes
        # application model naming.
        if "efficientnetb0" in layer_name:
            return layer

        if "mobilenetv2" in layer_name:
            return layer

        if "resnet50v2" in layer_name:
            return layer

    raise ValueError(
        "Could not find a supported pretrained backbone "
        "inside the model. Expected EfficientNetB0, "
        "MobileNetV2 or ResNet50V2."
    )


def unfreeze_top_n(
    model: keras.Model,
    n: int = 30,
) -> keras.Model:
    """
    Unfreeze the final n layers of the pretrained backbone.

    BatchNormalization layers remain frozen.

    The model must be recompiled after this function is called.
    """

    if n <= 0:
        raise ValueError(
            "n must be greater than 0."
        )

    backbone = find_backbone(model)

    total_layers = len(backbone.layers)

    if n > total_layers:
        print(
            f"Requested {n} layers, but the backbone "
            f"contains only {total_layers} layers."
        )

        n = total_layers

    # Enable the backbone before choosing individual layers.
    backbone.trainable = True

    # Freeze all layers first.
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze only the last n layers, except BatchNorm layers.
    for layer in backbone.layers[-n:]:
        if isinstance(
            layer,
            layers.BatchNormalization,
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_layer_count = sum(
        1
        for layer in backbone.layers
        if layer.trainable
    )

    trainable_parameter_count = sum(
        int(tf.size(variable))
        for variable in backbone.trainable_variables
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 60)
    print(f"Backbone found: {backbone.name}")
    print(f"Total backbone layers: {total_layers}")
    print(f"Requested final layers: {n}")
    print(
        "Actually trainable backbone layers: "
        f"{trainable_layer_count}"
    )
    print(
        "Trainable backbone parameters: "
        f"{trainable_parameter_count:,}"
    )
    print("BatchNormalization layers: frozen")
    print("=" * 60)

    return model