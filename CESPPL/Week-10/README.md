# Week 10 – Training, Fine-Tuning and Error Analysis

## Objective

The objective of Week 10 was to improve the CESPPL waste management activity classifier through fine-tuning, evaluate its performance, and perform detailed error analysis to better understand model behavior.

---

## Tasks Completed

### Model Fine-Tuning
- Fine-tuned the EfficientNetB0-based classifier using the CESPPL dataset.
- Compared the latest fine-tuned model with previous training runs.
- Selected the best-performing model for evaluation.

### Model Evaluation
- Evaluated the model on the validation dataset.
- Generated:
  - Confusion Matrix
  - Per-Class Metrics
  - Evaluation Metrics
  - Prediction Results
- Achieved:
  - **Validation Accuracy:** 95.03%
  - **Macro F1 Score:** 93.31%

### Error Analysis
- Identified the top confusion pairs from the confusion matrix.
- Displayed misclassified validation images.
- Categorized errors into:
  - Reasonable ambiguity
  - Model weakness
  - Label inconsistency
- Created summary tables for error categories.

### Grad-CAM Analysis
- Applied Grad-CAM to misclassified validation images.
- Visualized the regions used by the model for prediction.
- Observed whether the model focused on:
  - Workers
  - Waste bins
  - Cleaning equipment
  - Vehicles
  - Background regions
- Recorded observations and generated Grad-CAM analysis reports.

### Summary and Documentation
- Compared model performance across different runs.
- Documented strengths and weaknesses of the classifier.
- Prepared Week 10 summary and supporting analysis files.

---

## Key Learnings

- Fine-tuning improved overall classification performance.
- Confusion matrix analysis helped identify the most challenging activity pairs.
- Grad-CAM provided insight into the model's decision-making process.
- Most prediction errors occurred between visually similar waste management activities.
- Error analysis is essential for identifying whether improvements require better training strategies or additional data.

---

## Files Generated

- `w10_results_analysis.ipynb`
- `w10_error_analysis.ipynb`
- `w10_gradcam_analysis.ipynb`
- `evaluation_metrics.json`
- `per_class_metrics.csv`
- `predictions.csv`
- `confusion_matrix.png`
- `top_confusion_pairs.csv`
- `gradcam_analysis.csv`
- `WEEK10_SUMMARY.md`

---

## Future Improvements

- Collect additional samples for confusing activity classes.
- Improve data augmentation techniques.
- Reduce background bias during training.
- Experiment with advanced backbones and learning-rate schedules.
- Continue improving classifier robustness in Week 11.

---

**Week Status:** ✅ Completed
