# CESPPL Activity Classification Model Card

## Model Details

- **Model name:** CESPPL Activity Classification Model
- **Model version:** Week 11 canonical model
- **Model format:** Keras `.keras`
- **Architecture:** EfficientNetB0 transfer-learning classifier
- **Input size:** 224 × 224 × 3 RGB image
- **Output classes:** 10
- **Final checkpoint:** `runs/run_07_canonical/best.weights.h5`
- **Deployment model:** `models/final_model.keras`
- **Random seed:** 42
- **Fine-tuning:** Final 30 backbone layers considered, with BatchNormalization layers kept frozen
- **Training augmentation:** Horizontal flip, rotation, zoom, contrast and brightness augmentation
- **Framework:** TensorFlow/Keras
- **Model card structure:** Based on the model-card framework described by Mitchell et al. (2018)

## Class Labels

1. BIN LIFTING
2. BIN WASHING
3. GATE MEETING
4. LFC
5. MANUAL BEACH CLEANING
6. MECHANICAL SWEEPING
7. MECHANIZED BEACH CLEANING
8. PRIMARY COLLECTION
9. ROAD SWEEPING
10. SECONDARY VEHICLES

## Intended Use

The model is intended for internal CESPPL operational documentation.

It can assist with:

- Categorising field photographs into the ten covered operational activities.
- Organising operational evidence and activity records.
- Supporting internal dashboards and activity summaries.
- Reducing the manual effort required to sort large collections of field images.

The model should be used as a decision-support and documentation tool. Predictions, especially low-confidence predictions, should remain reviewable by an authorised user.

## Out-of-Scope Uses

The model is not intended for:

- Personnel evaluation or employee-performance scoring.
- Attendance monitoring.
- Disciplinary decisions.
- Safety-critical decision-making.
- Identifying individual workers.
- Facial recognition.
- Use outside the geographic and operational conditions represented by the CESPPL dataset.
- Classifying activities that are not among the ten trained classes.
- Fully autonomous evidence approval without human review.

## Training and Evaluation Data

The source dataset consisted of field photographs collected for CESPPL operational activities.

After preprocessing and identical-image deduplication:

- **Processed images:** 3,616
- **Number of classes:** 10
- **Split strategy:** Stratified train, validation and test split
- **Approximate split ratio:** 70% training, 15% validation and 15% test
- **Test images:** 543

Images were converted to RGB, resized for model input and grouped according to the operational class labels.

The dataset is imbalanced. Some activities, especially MANUAL BEACH CLEANING, have substantially more images than smaller classes such as LFC and PRIMARY COLLECTION.

## Final Configuration

- **Backbone:** EfficientNetB0
- **Image size:** 224 × 224
- **Dropout:** 0.3
- **Fine-tuning learning rate:** 1e-5
- **Unfrozen region:** Top 30 backbone layers
- **BatchNormalization:** Frozen
- **Augmentation mode:** Strong lighting augmentation
- **Canonical run:** `runs/run_07_canonical`
- **Seed:** 42

## Validation Metrics

- **Validation accuracy:** 95.58%
- **Validation macro-F1:** 93.30%

## Test Metrics

- **Test accuracy:** 95.03%
- **Test macro-F1:** 93.20%
- **Test samples:** 543

## Test Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BIN LIFTING | 1.0000 | 1.0000 | 1.0000 | 25 |
| BIN WASHING | 0.9231 | 0.9796 | 0.9505 | 49 |
| GATE MEETING | 0.9623 | 0.9444 | 0.9533 | 54 |
| LFC | 0.8750 | 0.7000 | 0.7778 | 20 |
| MANUAL BEACH CLEANING | 0.9659 | 0.9950 | 0.9802 | 199 |
| MECHANICAL SWEEPING | 1.0000 | 0.9259 | 0.9615 | 27 |
| MECHANIZED BEACH CLEANING | 0.9677 | 0.9677 | 0.9677 | 31 |
| PRIMARY COLLECTION | 0.8421 | 1.0000 | 0.9143 | 16 |
| ROAD SWEEPING | 0.9231 | 0.9231 | 0.9231 | 78 |
| SECONDARY VEHICLES | 0.9487 | 0.8409 | 0.8916 | 44 |

## CPU Inference Performance

- **Device:** User's Windows laptop CPU
- **Benchmark size:** 20 images
- **Average latency per image:** REPLACE_WITH_MEASURED_VALUE ms
- **Model loading:** Excluded from the per-image latency
- **Warm-up inference:** Performed before timing

This number should be replaced with the value printed by `predict.py`.

## Ethical Considerations

The dataset may contain identifiable workers, vehicles, locations and operational surroundings.

The following safeguards are recommended:

- Restrict access to authorised personnel.
- Avoid displaying identifiable worker images unnecessarily.
- Do not use predictions for personnel evaluation or disciplinary action.
- Store images and prediction records securely.
- Apply appropriate retention and access-control policies.
- Provide human review for low-confidence or disputed predictions.
- Avoid publishing raw field photographs without appropriate permission.

## Limitations and Caveats

### Small-class performance

Metrics for classes with few test examples have higher uncertainty. A small number of mistakes can change recall substantially.

LFC is the weakest class in the final test evaluation:

- Precision: 87.50%
- Recall: 70.00%
- F1-score: 77.78%
- Support: 20 images

This result should be interpreted cautiously because the class has limited test support.

### Lighting conditions

Strong brightness and contrast augmentation was used to improve robustness. However, extremely dark, overexposed or colour-distorted images may still produce unreliable predictions.

### Overlay sensitivity

Timestamps, watermarks, application overlays and large text regions may influence predictions if they cover important visual evidence.

### Geographic and operational limits

The model was trained using the available CESPPL field photographs. It has not been validated for different organisations, cities, camera systems or operational procedures.

### Unknown activities

The classifier always chooses one of its ten known classes. It does not currently contain an “unknown” or “other” class. Images outside the intended classes may therefore receive an incorrect but confident prediction.

### Confidence interpretation

Confidence is a model probability, not a guarantee of correctness. Low-confidence predictions and operationally important evidence should be manually reviewed.

## Evaluation Artifacts

The canonical test artifacts are stored in:

`final_results/`

This directory includes:

- `evaluation_metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.png`
- `predictions.csv`

## Deployment Files

The serving system depends on:

- `models/final_model.keras`
- `models/class_names.json`
- `src/classifier.py`

Both the command-line predictor and the future web application must call `src/classifier.py` rather than implementing separate preprocessing logic.