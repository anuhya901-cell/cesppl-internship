import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


SUPPORTED_BACKBONES = [
    "efficientnetb0",
    "mobilenetv2",
    "resnet50v2",
]


def create_augmentation():
    """Create image augmentation layers used during normal training."""

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def get_backbone(
    backbone_name,
    img_size,
    trainable=False,
):
    """Create the selected ImageNet backbone."""

    backbone_name = backbone_name.lower().strip()

    if backbone_name == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )

        preprocess_function = (
            keras.applications.efficientnet.preprocess_input
        )

    elif backbone_name == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )

        preprocess_function = (
            keras.applications.mobilenet_v2.preprocess_input
        )

    elif backbone_name == "resnet50v2":
        backbone = keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )

        preprocess_function = (
            keras.applications.resnet_v2.preprocess_input
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from {SUPPORTED_BACKBONES}."
        )

    backbone.trainable = trainable

    return backbone, preprocess_function


def build_model(
    backbone_name="efficientnetb0",
    num_classes=10,
    img_size=224,
    dropout=0.3,
    learning_rate=1e-3,
    use_augmentation=True,
    backbone_trainable=False,
):
    """
    Build and compile the CESPPL image classifier.

    Parameters
    ----------
    backbone_name:
        efficientnetb0, mobilenetv2 or resnet50v2.

    use_augmentation:
        True for normal training.
        False for sanity-overfit and evaluation.

    backbone_trainable:
        False for normal feature-extraction training.
        True during the sanity-overfit test.
    """

    backbone_name = backbone_name.lower().strip()

    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from {SUPPORTED_BACKBONES}."
        )

    backbone, preprocess_function = get_backbone(
        backbone_name=backbone_name,
        img_size=img_size,
        trainable=backbone_trainable,
    )

    inputs = keras.Input(
        shape=(img_size, img_size, 3),
        name="input_image",
    )

    x = inputs

    if use_augmentation:
        augmentation = create_augmentation()
        x = augmentation(x)

    x = layers.Lambda(
        preprocess_function,
        name=f"{backbone_name}_preprocessing",
    )(x)

    x = backbone(
        x,
        training=backbone_trainable,
    )

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        dropout,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"cesppl_{backbone_name}",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model