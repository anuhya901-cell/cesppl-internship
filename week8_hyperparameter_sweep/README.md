# Week 8 Hyperparameter Sweep

This folder contains the hyperparameter sweep notebooks and experiment results for the TrashNet project.

# Week 8 – Hyperparameter Sweep and Iteration Playbook

## Project Overview

The objective of this week's work was to understand how different hyperparameters affect the performance of an image classification model rather than simply searching for the highest possible accuracy.

A systematic fractional hyperparameter sweep was performed using the TrashNet dataset and the EfficientNetB0 transfer learning model. Only three hyperparameters were changed while keeping the rest of the training pipeline unchanged to ensure a fair comparison.

---

## Hyperparameters Investigated

### Image Size
- 160 × 160
- 224 × 224
- 260 × 260

### Dropout Rate
- 0.2
- 0.3
- 0.5

### Data Augmentation

**Low**
- Horizontal Flip
- Rotation

**Medium**
- Horizontal Flip
- Rotation
- Zoom
- Contrast

**High**
- Horizontal Flip
- Rotation
- Zoom
- Contrast
- Brightness

---

## Experimental Design

Instead of performing all 27 possible combinations (3 × 3 × 3), a six-run fractional design was selected to reduce computational cost while covering representative combinations.

| Run | Image Size | Dropout | Augmentation |
|------|-----------|----------|--------------|
| Run 1 | 160 | 0.2 | Low |
| Run 2 | 160 | 0.5 | High |
| Run 3 | 224 | 0.3 | Medium |
| Run 4 | 224 | 0.5 | Low |
| Run 5 | 260 | 0.2 | High |
| Run 6 | 260 | 0.5 | Medium |

---

## Backbone

- EfficientNetB0
- ImageNet Pretrained
- Feature Extraction Pipeline

---

## Common Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- EarlyStopping
- ReduceLROnPlateau

---

## Results

The complete experimental results are available in:

**experiments.csv**

The file records:

- Training Accuracy
- Validation Accuracy
- Training Time
- Hyperparameter Configuration
- Experiment Notes

---

## Key Observations

- Smaller image sizes reduced training time but could slightly reduce classification performance.
- Increasing dropout improved regularization but excessive dropout occasionally reduced accuracy.
- Medium augmentation generally produced more stable validation performance.
- High augmentation increased robustness but sometimes slowed convergence.
- The hyperparameter sweep confirmed that carefully selected parameters provide a good balance between accuracy and training efficiency.

---

## Skills Demonstrated

- Hyperparameter Optimization
- Experimental Design
- Transfer Learning
- EfficientNetB0
- Image Classification
- Deep Learning
- Data Augmentation
- Model Evaluation
- Experiment Tracking

---

## Conclusion

This week's work demonstrated the importance of systematic experimentation instead of random parameter tuning. By changing only a few hyperparameters at a time, it became possible to understand their individual effects on model performance and establish a repeatable experimentation workflow for future deep learning projects.

## Final Sweep Result

Best of six sweep runs: **89.31% validation accuracy** with:

- Image size: **260 × 260**
- Dropout: **0.5**
- Augmentation: **Medium**
- Training time: **4.86 minutes**

The sweep suggested that image size had the clearest effect on validation accuracy, while moderate augmentation provided a better balance than very strong augmentation.

