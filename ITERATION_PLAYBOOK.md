# Model Iteration Playbook

## Purpose

This playbook is a practical troubleshooting guide for training image-classification models on the CESPPL dataset and other future datasets.

Its purpose is to provide a clear sequence of checks and experiments when model training does not work as expected. Instead of changing many settings randomly, this playbook helps diagnose problems systematically, record results, and make decisions based on evidence.

---

# 1. Pre-Training Checks

Before training any new model, complete the following checks in order.

## 1.1 Verify the Dataset Folder Structure

Confirm that the dataset follows the required folder structure:

```text
dataset/
├── class_1/
├── class_2/
├── class_3/
└── ...
```

Each class must have its own folder, and every image must be placed inside the correct class folder.

Check that:

- There are no images directly inside the dataset root.
- Folder names are spelled consistently.
- No duplicate class folders exist.
- No temporary files or unrelated files are present.
- Every expected class is included.

## 1.2 Compare Dataset Classes with `CLASSES.md`

Load the dataset using `image_dataset_from_directory()` and print the detected class names.

```python
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32
)

print(train_dataset.class_names)
```

Compare this list with `CLASSES.md`.

The order and names must match exactly. If the folder names differ from the expected labels, fix the dataset folders before training.

## 1.3 Display Ten Images Per Class

Display at least ten images from every class.

Check for:

- Incorrect labels
- Blurry images
- Very dark or overexposed images
- Corrupted images
- Objects that are too small
- Images dominated by the background
- Duplicate or nearly identical images
- Images containing multiple waste categories

Do not assume the dataset is correct simply because it loads without errors.

## 1.4 Confirm Split Sizes

Print the number of images in the training, validation, and test sets.

Example:

```python
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))
print("Test batches:", len(test_dataset))
```

Also verify the class labels in each split:

```python
train_labels = np.concatenate(
    [labels.numpy() for _, labels in train_dataset]
)

validation_labels = np.concatenate(
    [labels.numpy() for _, labels in validation_dataset]
)

print("Training labels:", np.unique(train_labels))
print("Validation labels:", np.unique(validation_labels))
```

All expected classes must be present in every required split.

## 1.5 Check Class Distribution

Count the number of images in every class.

```python
class_counts = np.bincount(
    train_labels,
    minlength=len(class_names)
)

for index, count in enumerate(class_counts):
    print(class_names[index], count)
```

If one class is much smaller than the others, plan to evaluate:

- Class weighting
- Oversampling
- Additional data collection
- Per-class precision and recall

## 1.6 Confirm Image Properties

Check:

- Image dimensions
- Number of colour channels
- Data type
- Value range
- Colour format

Example:

```python
for images, labels in train_dataset.take(1):
    print("Shape:", images.shape)
    print("Data type:", images.dtype)
    print("Minimum value:", tf.reduce_min(images).numpy())
    print("Maximum value:", tf.reduce_max(images).numpy())
```

The model input shape and preprocessing must match the actual image data.

---

# 2. Overfit-One-Batch Sanity Check

Before full training, try to make the model overfit one small batch.

The purpose of this check is to verify that:

- The model can learn.
- Labels align with images.
- The loss function is correct.
- The output layer matches the number of classes.
- The optimizer is updating weights.
- The preprocessing pipeline is valid.

Example:

```python
small_batch = train_dataset.take(1)

history = model.fit(
    small_batch,
    epochs=50,
    verbose=1
)
```

The model should eventually reach very high accuracy on that single batch.

If it cannot overfit one batch, do not start full training. Check:

- Wrong output activation
- Wrong loss function
- Incorrect labels
- Incorrect preprocessing
- Frozen layers that should be trainable
- Learning rate too small
- Dataset corruption
- Model output size not equal to the number of classes

A failed one-batch test usually indicates a pipeline problem rather than insufficient model capacity.

---

# 3. First-Pass Model

Use the following configuration as the default starting point for a new image-classification dataset.

## Architecture

- Backbone: EfficientNetB0
- Pretrained weights: ImageNet
- Backbone initially frozen
- GlobalAveragePooling2D
- Dropout: 0.3
- Dense output layer with `num_classes`
- Output activation: Softmax

## Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])
```

## Model

```python
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)
```

## Compilation

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Callbacks

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)
```

## First Training Run

```python
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)
```

Do not change several parameters at once during the first run. The first objective is to establish a reliable baseline.

---

# 4. If Accuracy Is Below 60%

Accuracy below 60% after the first pass usually indicates a pipeline or data problem.

Perform these checks before trying a larger model.

## Check 1: Verify the Preprocessing Function

Use the preprocessing function that matches the backbone.

For EfficientNetB0:

```python
tf.keras.applications.efficientnet.preprocess_input
```

For DenseNet121:

```python
tf.keras.applications.densenet.preprocess_input
```

Do not use preprocessing from a different architecture.

## Check 2: Verify the Image Size

Confirm that:

- The dataset loader uses the intended image size.
- The model input shape uses the same image size.
- The pretrained backbone supports that size.

Example:

```python
IMAGE_SIZE = 224
```

Use the same value consistently in dataset loading and model construction.

## Check 3: Check Class Labels and Folder Names

Print:

```python
print(class_names)
```

Compare the result against `CLASSES.md`.

Inspect random images together with their labels. Correct any mislabeled folders or images.

## Check 4: Confirm Train and Validation Sets Are Correct

Verify that training and validation data were not swapped.

Check:

```python
print(len(train_dataset))
print(len(validation_dataset))
```

The training split should normally be larger than the validation split.

Also verify that all classes appear in both splits.

## Check 5: Run the Overfit-One-Batch Test

If the model cannot learn one small batch, stop full training.

Recheck:

- Loss function
- Output layer
- Label format
- Learning rate
- Trainable parameters
- Preprocessing

Do not continue running long experiments until this test succeeds.

---

# 5. If Accuracy Is Between 60% and 75%

Accuracy in this range usually means the pipeline is working, but the model is undertrained or the setup needs moderate improvement.

Try the following one at a time.

## 5.1 Train Longer

Increase the maximum number of epochs while keeping EarlyStopping enabled.

Example:

```python
epochs=30
```

Do not remove EarlyStopping.

## 5.2 Reduce Augmentation Strength

Strong augmentation may make a small dataset unnecessarily difficult.

Temporarily use:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05)
])
```

Compare the validation curves before deciding.

## 5.3 Check Class Imbalance

Compute class counts and inspect per-class recall.

If minority classes perform poorly, try:

```python
class_weight=class_weight
```

inside `model.fit()`.

Do not judge the model only by overall accuracy.

## 5.4 Try a Controlled Hyperparameter Change

Change only one major setting at a time:

- Image size
- Dropout
- Augmentation
- Learning rate

Record every experiment in `experiments.csv`.

Avoid changing several settings simultaneously because it becomes impossible to identify what caused the result.

---

# 6. Fine-Tuning Procedure

Fine-tuning should begin only after feature extraction produces a stable baseline.

## 6.1 Unfreeze the Top Thirty Layers

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True
```

## 6.2 Keep the Base Model in Inference Mode

The backbone must still be called with:

```python
x = base_model(
    x,
    training=False
)
```

This prevents Batch Normalization statistics from changing aggressively on a small dataset.

## 6.3 Recompile with a Smaller Learning Rate

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Always recompile after changing layer trainability.

## 6.4 Fine-Tune for Up to Fifteen Epochs

```python
fine_tune_history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)
```

Compare fine-tuning results against the feature-extraction baseline.

Fine-tuning is not automatically better. Keep the stronger checkpoint even when it comes from feature extraction.

---

# 7. Common Mistakes from This Programme

## Mistake 1: Forgetting the Correct `preprocess_input`

Using the wrong preprocessing function can significantly reduce model performance.

Always match preprocessing to the chosen backbone.

## Mistake 2: Using the Wrong Colour Space

OpenCV loads images in BGR order by default, while TensorFlow and Matplotlib normally use RGB.

When using OpenCV, convert correctly:

```python
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

## Mistake 3: Training Without Callbacks

Running `model.fit()` without EarlyStopping or a learning-rate strategy can waste time and increase overfitting.

Always use callbacks unless the experiment specifically tests a different schedule.

## Mistake 4: Reporting Accuracy Without a Confusion Matrix

Overall accuracy can hide poor minority-class performance.

Always generate:

- Confusion matrix
- Classification report
- Per-class precision
- Per-class recall
- Per-class F1-score

## Mistake 5: Using Vertical Flip on Natural Images

Vertical flipping is often unrealistic for natural object images.

Waste objects, animals, people, and most real-world scenes are not normally upside down.

Use horizontal flip unless vertical orientation is genuinely irrelevant to the dataset.

---

# 8. When to Stop

Stop experimenting when:

- Validation accuracy has not improved for ten epochs.
- The same plateau occurs across two consecutive training runs.
- The two runs use different reasonable learning rates.
- Training curves remain stable.
- The remaining errors are mainly ambiguous images, poor-quality images, or questionable labels.
- Additional model complexity gives only a very small improvement relative to training time.

A practical stopping rule is:

> Stop when validation accuracy fails to improve for ten epochs across two consecutive runs using different learning rates.

Before stopping, confirm that:

- The dataset has been inspected carefully.
- The one-batch sanity check passed.
- At least one feature-extraction and one fine-tuning run were completed.
- Class imbalance was evaluated.
- Confusion pairs were inspected.
- The strongest model was saved.
- Results were recorded in `experiments.csv`.
- The README and learning log were updated.

The goal is not to run experiments forever. The goal is to reach a stable, well-understood model and document why further tuning is unlikely to provide meaningful value.

---

# Final Reminder

When a model performs poorly:

1. Check the data.
2. Check the labels.
3. Check preprocessing.
4. Run the one-batch sanity test.
5. Establish a simple baseline.
6. Change one variable at a time.
7. Read the training curves.
8. Inspect the confusion matrix.
9. Study misclassified images.
10. Record every experiment.

Do not panic and do not change the entire pipeline at once. Diagnose the problem methodically.

# Personal Lessons from the TrashNet Project

Throughout the TrashNet project, I learned several practical lessons that I will apply to future datasets.

- Always inspect the dataset before training.
- Verify class labels before starting experiments.
- Compare feature extraction and fine-tuning fairly.
- Record every experiment in experiments.csv.
- Analyse confusion matrices instead of relying only on accuracy.
- Study misclassified images before changing the model.
- Save the best model checkpoint after every successful experiment.
- Change only one major hyperparameter at a time.
- Document every experiment so results remain reproducible.
