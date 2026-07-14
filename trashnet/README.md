# # TrashNet Image Classification Project

This folder contains all Week 6 and Week 7 TrashNet notebooks and project files.

# Week 6 & Week 7 – TrashNet Image Classification

## Project Overview

During Weeks 6 and 7, I developed a complete deep learning image classification pipeline using the **TrashNet** dataset. The project focused on transfer learning, fine-tuning, model evaluation, error analysis, backbone comparison, and handling class imbalance.

The goal was to build an efficient waste classification model capable of identifying six waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## Dataset Split

- Total Images: **2,527**
- Training Images: **2,022**
- Validation Images: **505**
- Number of Classes: **6**

# Week 6 – TrashNet Feature Extraction & Initial Fine-Tuning

## Objectives

- Explore the TrashNet dataset
- Train an EfficientNetB0 transfer learning model
- Perform feature extraction
- Conduct the first fine-tuning experiment
- Evaluate model performance

### Model

- Backbone: EfficientNetB0
- Pre-trained on ImageNet
- Input Size: 224 × 224
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

---

## Feature Extraction Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **91.89%** |
| Validation Accuracy | **88.12%** |

The feature extraction model achieved strong validation performance and served as the baseline model for subsequent experiments.

---

## First Fine-Tuning Results

Top 30 layers of EfficientNetB0 were unfrozen and trained with a lower learning rate.

| Metric | Value |
|---------|-------|
| Training Accuracy | **86.55%** |
| Validation Accuracy | **85.94%** |

---

# Week 7 – Advanced Model Improvement & Analysis

Week 7 focused on improving the model and understanding its prediction behaviour through multiple experiments.

---

## 1. Proper Fine-Tuning

The EfficientNetB0 model was fine-tuned using a lower learning rate and additional epochs.

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **86.80%** |
| Validation Accuracy | **85.74%** |

---

## 2. Error Analysis

Performed detailed model analysis using:

- Confusion Matrix
- Classification Report
- Misclassified Image Inspection
- Grad-CAM Visualization (representative samples)

### Most Common Confusion Pairs

- Plastic → Glass
- Plastic → Metal
- Plastic → Trash

### Key Observations

- Transparent plastic objects often resemble glass.
- Reflective plastic surfaces can appear similar to metal.
- Damaged plastic objects are sometimes classified as general trash.
- Most prediction errors were caused by visual similarity rather than model failure.

---

## 3. DenseNet121 Backbone Comparison

A second transfer learning model was implemented to compare backbone performance.

### DenseNet121 Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **83.53%** |
| Validation Accuracy | **86.34%** |

### Backbone Comparison

| Backbone | Best Validation Accuracy |
|-----------|-------------------------:|
| EfficientNetB0 | **88.12%** |
| DenseNet121 | **86.34%** |

**Selected Backbone:** EfficientNetB0

EfficientNetB0 provided the best overall validation performance while maintaining stable training.

---

## 4. Class Weighting for Imbalanced Data

To improve minority-class performance, class weights were applied during training.

### Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **91.20%** |
| Restored Best Validation Accuracy | **87.72%** |

### Impact

| Model | Validation Accuracy |
|--------|--------------------:|
| Without Class Weights | **88.12%** |
| With Class Weights | **87.72%** |

Although overall validation accuracy decreased slightly, the recall of the minority **Trash** class improved.

| Metric | Before | After |
|---------|-------:|------:|
| Trash Recall | **0.74** | **0.82** |

This demonstrated the effectiveness of class weighting for handling class imbalance.

---

---

# Hyperparameter Sweep

A six-run fractional hyperparameter sweep was performed to study how different training settings affect model performance.

### Hyperparameters Tested

- Image Size: 160, 224, 260
- Dropout: 0.2, 0.3, 0.5
- Augmentation:
  - Low
  - Medium
  - High

### Best Sweep Run

| Setting | Value |
|----------|-------|
| Run | **6** |
| Image Size | **260 × 260** |
| Dropout | **0.5** |
| Augmentation | **Medium** |
| Validation Accuracy | **89.31%** |
| Training Time | **4.86 minutes** |

### Key Observation

The hyperparameter sweep showed that larger image sizes generally produced better validation accuracy, while medium augmentation provided a better balance than aggressive augmentation.

---

# Cosine Learning-Rate Schedule

A cosine learning-rate schedule was evaluated using the best-performing sweep configuration.

| Training Strategy | Validation Accuracy |
|-------------------|--------------------:|
| Constant Learning Rate | **89.31%** |
| Cosine Decay | **XX.XX%** |

> Replace **XX.XX%** with your measured cosine-decay validation accuracy after completing the notebook.

### Observation

The cosine-decay experiment was performed to determine whether a gradually decreasing learning rate improved convergence compared to a constant learning rate.

# Final Selected Model

After evaluating feature extraction, fine-tuning, DenseNet121, class weighting, and hyperparameter tuning, the final selected configuration is:

- Backbone: **EfficientNetB0**
- Image Size: **260 × 260**
- Dropout: **0.5**
- Augmentation: **Medium**
- Optimizer: **Adam**
- Best Validation Accuracy: **89.31%**

This configuration will serve as the starting point for the real CESPPL operational dataset.

---

# Techniques Used

- TensorFlow / Keras
- Transfer Learning
- EfficientNetB0
- DenseNet121
- Fine-Tuning
- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Confusion Matrix
- Classification Report
- Error Analysis
- Grad-CAM
- Class Weighting
- Model Comparison

---

# Key Learning Outcomes

- Built an end-to-end image classification pipeline using transfer learning.
- Understood the impact of fine-tuning on model performance.
- Compared different CNN backbone architectures.
- Learned to interpret confusion matrices and classification reports.
- Performed systematic error analysis using misclassified images.
- Applied class weighting to improve minority-class recognition.
- Developed a structured workflow for evaluating and improving deep learning models.
- - Designed and executed a structured hyperparameter sweep.
- Compared multiple image sizes, dropout values, and augmentation strategies.
- Evaluated the effect of learning-rate scheduling using cosine decay.
- Documented experiments using experiments.csv.
- Established a reproducible workflow for future CESPPL experiments.
