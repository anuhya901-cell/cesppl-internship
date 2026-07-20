# CESPPL Dataset Card

## Dataset Summary

The processed CESPPL dataset contains **3616 images**
across **10 operational activity classes**.

The frozen dataset split uses:

- Training images: **2530**
- Validation images: **543**
- Test images: **543**
- Random seed: **42**
- Split strategy: **stratified 70/15/15 split**

## Purpose

To support the development and evaluation of computer-vision models that classify CESPPL operational activities from field photographs.

The dataset is intended to help study whether operational activities
can be distinguished using image-classification methods.

## Dataset Content

The dataset contains the following classes:

- **BIN_LIFTING**: 166 images
- **BIN_WASHING**: 325 images
- **GATE_MEETING**: 360 images
- **LFC**: 135 images
- **MANUAL_BEACH_CLEANING**: 1328 images
- **MECHANICAL_SWEEPING**: 179 images
- **MECHANIZED_BEACH_CLEANING**: 204 images
- **PRIMARY_COLLECTION**: 109 images
- **ROAD_SWEEPING**: 518 images
- **SECONDARY_VEHICLES**: 292 images

All processed images were corrected for EXIF orientation, converted to
RGB, resized and centre-cropped to **320 × 320 pixels**, and saved as
JPEG images.

Identical perceptual-hash duplicates were removed before creating the
train, validation and test splits.

## Collection Process

The images were captured by field staff using mobile phones during
real CESPPL operational activities.

**Collection period:** Collected during CESPPL field operations; exact collection dates should be confirmed with the operations team.

**Collection locations:** CESPPL operational locations, including roads, beaches, collection points, depots, gates and waste-management activity areas.

The dataset includes photographs of roads, beaches, vehicles, bins,
workers, depots, gates and waste-management environments.

## Labelling Process

The images were labelled through folder placement by the CESPPL
operations team.

Each folder represents one operational activity. The folder name is
used as the class label.

The labels were visually reviewed during Week 9 to identify possible
mislabelled, unclear, low-quality or multi-activity images.

## Dataset Splits

The dataset was split once using
`sklearn.model_selection.train_test_split`.

The split was stratified by class and used `random_state=42`.

The resulting files are:

- `train.csv`
- `val.csv`
- `test.csv`

These split files are frozen and should be reused for all future
experiments. They should not be regenerated between model runs.

## Class Balance

The largest class is **MANUAL_BEACH_CLEANING** with
**1328 images.

The smallest class is **PRIMARY_COLLECTION** with
**109 images.

The largest-to-smallest class ratio is approximately
**12.18:1**.

This imbalance may cause the model to perform better on common classes
and less consistently on smaller classes.

## Known Limitations

- The dataset is imbalanced across the ten classes.
- Some images contain timestamps, GPS overlays, application overlays or
  camera watermarks.
- Overlay styles may be associated with particular phones, staff
  members or classes and could become shortcut features.
- Lighting varies between daylight, early morning, low-light and night
  conditions.
- Images may contain motion blur, unusual camera angles or partial
  views of the activity.
- Some visually similar classes share the same workers, roads, bins,
  beaches and vehicles.
- Some photographs may show more than one activity in the same frame.
- The dataset represents a specific operational environment and may not
  generalise to other organisations, cities or waste-management systems.
- Small test classes contain approximately 20 images, so one prediction
  may change class recall by about five percentage points.

## Ethical Considerations

Workers may be identifiable in some images.

The images should remain internal to the authorised project and should
not be publicly released without organisational approval and an
appropriate privacy review.

The dataset should not be used for facial recognition, worker
identification, employee monitoring or individual performance
assessment.

## Intended Uses

- Internal research on operational activity classification.
- Training and evaluating image-classification models.
- Studying class imbalance and visual similarity between operational
  activities.
- Supporting future internal waste-management automation research.

## Out-of-Scope Uses

- Facial recognition or worker identification.
- Surveillance or employee performance monitoring.
- Public release without permission.
- Use in unrelated organisations without validation.
- Safety-critical decisions without human review.
- Inferring sensitive information about workers.

## Maintenance

The train, validation and test splits are frozen.

Future experiments must read the filenames from the saved CSV files
instead of generating new random splits.
