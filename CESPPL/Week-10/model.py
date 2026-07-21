import tensorflow as tf


SUPPORTED_BACKBONES = [
    "efficientnetb0",
    "mobilenetv2",
    "resnet50v2"
]


def build_model(
    num_classes,
    backbone="efficientnetb0",
    img_size=224,
    dropout=0.3,
    learning_rate=1e-3
):
    backbone = backbone.lower().strip()

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone}"
        )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation"
    )

    input_shape = (img_size, img_size, 3)

    if backbone == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        preprocess = (
            tf.keras.applications.efficientnet.preprocess_input
        )

    elif backbone == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        preprocess = (
            tf.keras.applications.mobilenet_v2.preprocess_input
        )

    else:
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        preprocess = (
            tf.keras.applications.resnet_v2.preprocess_input
        )

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=input_shape,
        name="input_image"
    )

    x = augmentation(inputs)

    x = tf.keras.layers.Lambda(
        preprocess,
        name=f"{backbone}_preprocessing"
    )(x)

    x = base_model(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        dropout
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name=f"cesppl_{backbone}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model