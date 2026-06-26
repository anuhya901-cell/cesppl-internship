![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)

![License](https://img.shields.io/badge/License-MIT-green) 


CESPPL Internship — Image Classification for Solid Waste Operations

This repository contains internship work related to image classification for solid waste operations using Python, machine learning, and computer vision tools.

# My ML Journey

Week 1 completed.
# Week 2 - Pandas Practice

In this week, I learned the basics of Pandas and Exploratory Data Analysis (EDA) using a real-world air pollution dataset from India.

## Topics Learned

- Creating and working with Pandas DataFrames and Series
- Loading CSV files using `pd.read_csv()`
- Understanding dataset structure using:
  - `shape`
  - `columns`
  - `head()`
- Selecting categorical and numeric columns
- Finding unique values using `nunique()`
- Calculating statistical measures such as:
  - mean
  - median
- Detecting missing values using:
  - `isnull()`
  - `sum()`
- Using `groupby()` for data aggregation
- Creating visualizations using Matplotlib:
  - bar charts
  - histograms
  - scatter plots
- Writing observations from exploratory data analysis

## Dataset Used

Air pollution monitoring dataset containing:
- states
- cities
- stations
- pollutant types
- pollution measurements

## Key Learning

I understood how Pandas helps in cleaning, analyzing, summarizing, and visualizing real-world datasets efficiently.



## Week 2 Progress — Fashion-MNIST CNN Classifier

Built and trained a Convolutional Neural Network (CNN) using TensorFlow and Keras on the Fashion-MNIST dataset.

### Project Highlights
- Loaded and preprocessed image data using TensorFlow
- Visualized clothing images from the dataset
- Built a CNN architecture with Conv2D, MaxPooling, Flatten, and Dense layers
- Trained the model for 10 epochs using the Adam optimizer
- Evaluated model performance on the test dataset
- Plotted training/validation accuracy and loss graphs
- Generated a confusion matrix for prediction analysis

### Model Performance
- Fashion-MNIST classifier reached 91.2% test accuracy.
- Training accuracy was higher than validation accuracy, indicating initial overfitting.
- The model performed well on visually distinct classes such as bags, sneakers, and ankle boots.
- Classes like shirts, pullovers, and T-shirts were more difficult because of similar visual patterns.

### Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

This project was my first deep learning image classification model and my first machine learning performance result published on GitHub.
                                                                                                                             Week 2 Results

MLP test accuracy: 85.4%
CNN test accuracy: 89.2%
CNN parameter count: 34,826


## Week 3: CIFAR-10 Baseline CNN

Built and trained a Convolutional Neural Network (CNN) from scratch on the CIFAR-10 dataset using TensorFlow/Keras. The model consists of four convolutional layers, two max-pooling layers, and two dense layers for image classification across 10 categories.

### Key Learning Outcomes

* Loaded and explored the CIFAR-10 dataset.
* Visualized random training images and their labels.
* Built a CNN architecture for image classification.
* Trained the model using the Adam optimizer.
* Monitored training and validation accuracy/loss curves.
* Observed overfitting by comparing training and validation performance.
* Evaluated the model on the test dataset.

### Results

* Test Accuracy: 70.08%
* Visible overfitting from approximately epoch Y onward.

### Technologies Used

* Python

* ## Week 3: CNN Overfitting and Data Augmentation

This week I trained a CNN on the CIFAR-10 dataset and observed clear signs of overfitting. I then applied data augmentation techniques such as horizontal flipping and random image transformations to improve generalization. The augmented model achieved a validation accuracy of 79.4% compared to 71.5% for the baseline model while using the same network architecture.

Baseline Validation Accuracy: 71.5%
Augmented Validation Accuracy: 79.4%

Baseline Test Accuracy: 70.8%
Augmented Test Accuracy: 78.9%
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab (GPU)

Week 4: Transfer Learning with MobileNetV2
Cats vs Dogs Feature Extraction
Loaded the Cats vs Dogs dataset from TensorFlow Datasets.
Applied image resizing and preprocessing.
Used MobileNetV2 pretrained on ImageNet as a frozen feature extractor.
Added a custom classification head with GlobalAveragePooling2D and Dropout.
Trained for 10 epochs using Transfer Learning.
Achieved 98.21% validation accuracy.
Saved trained model weights for future fine-tuning.

Result: Cats vs Dogs feature extraction: 98.21% validation accuracy in ten epochs.


## 🐱🐶 Cats vs Dogs Classification — Fine-Tuning Phase

This project extends the feature extraction model by applying *fine-tuning on MobileNetV2* to further improve performance on the Cats vs Dogs dataset.

---

## 🔧 Approach

- Loaded pretrained feature extraction model (cats_dogs_feature_extraction.keras)
- Used MobileNetV2 as base model
- Unfroze the last ~30 layers of the base model
- Applied a very small learning rate (*1e-5*) to preserve pretrained weights
- Continued training on the same dataset for additional epochs
- Evaluated performance after fine-tuning

---

## 📊 Results

| Phase | Validation Accuracy | Notes |
|------|---------------------|------|
| Feature Extraction | 98.21% | Trained only classification head |
| Fine-Tuning | 98.3% | Improved generalization by unfreezing top layers |

- Loss decreased further during fine-tuning
- Reduced overfitting compared to feature extraction phase
- Improved model stability and accuracy

---

## 🧠 Key Learning

Fine-tuning allows pretrained CNN models to adjust high-level features for a specific dataset. Using a very small learning rate (1e-5) is critical to avoid destroying learned ImageNet features.

---

## 💾 Saved Models

- cats_dogs_feature_extraction.keras
- cats_dogs_finetuned.keras

---

## ⚙️ Tools Used

- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- TensorFlow Datasets (Cats vs Dogs)

- ## Week 4 - Transfer Learning Comparison

This week I learned transfer learning using pretrained convolutional neural networks. Instead of training a model from scratch, I used pretrained ImageNet weights and adapted the models to the Cats vs Dogs classification task. I compared three different backbone architectures: MobileNetV2, ResNet50V2, and EfficientNetB0. For each model, I performed feature extraction followed by fine-tuning and evaluated validation accuracy, training speed, and model size.

| Backbone       | Parameters | Feature Extraction Accuracy | Fine-Tuning Accuracy | Time/Epoch | Weight Size |
| -------------- | ---------- | --------------------------- | -------------------- | ---------- | ----------- |
| MobileNetV2    | 2,257,985        | 98.36%                         | 98.2%                 | 30 sec        | 14 MB         |
| ResNet50V2     | 2,268,865        | 92.0%                         | 93.6%                  | 80 sec         | 98 MB         |
| EfficientNetB0 | 5,834,908         | 96.3%                          | 98.2%                   | 36 sec         | 22MB          |

### Takeaway

EfficientNetB0 achieved the best balance between accuracy and computational cost. MobileNetV2 was the fastest and smallest model, while ResNet50V2 provided a strong baseline but required more computation. This experiment demonstrated that model selection is an engineering trade-off rather than simply choosing the highest accuracy.


## Week 5: Flowers Dataset and Karpathy Sanity Check

Built a flower classification model using EfficientNetB0 transfer learning on the TensorFlow Flowers dataset containing five classes: daisy, dandelion, roses, sunflowers, and tulips.

Applied Andrej Karpathy's overfit-one-batch sanity check by training on only two images. The model successfully memorized the images, reducing the loss from 1.3561 to 0.0035 and achieving 100% accuracy. This verified that the data pipeline, model architecture, optimizer, and loss function were functioning correctly.

After passing the sanity check, trained the model on the full Flowers dataset and achieved:

- Training Accuracy: 100%
- Validation Accuracy: 100%
- Validation Loss: 0.0010

Key Learning:
The overfit-one-batch test is a powerful debugging technique that confirms whether a training pipeline is capable of fitting data before performing large-scale experiments.


# 🌸 Flowers Image Classification using MobileNetV2

## 📌 Project Overview

This project implements a Flower Image Classification model using Transfer Learning with MobileNetV2. The model is trained on the TensorFlow Flowers dataset containing five flower categories.

The pretrained MobileNetV2 base model is used as a frozen feature extractor, while custom classification layers are added on top for flower classification.

---

## 🎯 Objective

* Perform feature extraction using MobileNetV2.
* Apply data augmentation to improve generalization.
* Use callbacks to optimize training.
* Achieve validation accuracy above 88%.

---

## 📂 Dataset

The TensorFlow Flowers dataset contains the following classes:

* Daisy
* Dandelion
* Roses
* Sunflowers
* Tulips

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🔄 Data Augmentation

The following augmentation techniques were applied:

* RandomFlip
* RandomRotation (0.15)
* RandomZoom (0.15)
* RandomContrast (0.1)
* RandomTranslation (0.1, 0.1)

---

## 🧠 Model Architecture

* MobileNetV2 (Pretrained on ImageNet)
* Frozen Convolutional Base
* GlobalAveragePooling2D
* Dropout (0.3)
* Dense Layer (5 classes, Softmax)

---

## ⚙️ Training Configuration

* Optimizer: Adam (Learning Rate = 0.001)
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy
* Callbacks:

  * EarlyStopping
  * ReduceLROnPlateau

---

## 📊 Results

| Metric              | Value      |
| ------------------- | ---------- |
| Validation Accuracy | 89.51%     |
| Best Epoch          | 26         |
| Target Accuracy     | 88%        |
| Status              | Achieved ✅ |

---

## 📈 Confusion Matrix Analysis

* Dandelions achieved the highest classification accuracy.
* Roses and Tulips showed minor confusion due to visual similarity.
* Sunflowers were classified accurately with very few errors.
* Most predictions were concentrated along the diagonal, indicating strong model performance.

---

## 🎓 Key Learnings

* Transfer learning significantly reduces training time while maintaining strong performance.
* Data augmentation improves model generalization on smaller datasets.
* EarlyStopping prevents unnecessary training and reduces overfitting.
* ReduceLROnPlateau helps the model converge more effectively.

---

## 💾 Model Output

The trained model was saved in:

```text
flowers_feature_extraction.keras
```

---

## 🚀 Future Work

* Fine-tune the MobileNetV2 base model.
* Experiment with different learning rates.
* Compare performance with EfficientNet and ResNet architectures.
* Deploy the model as a web application.

## Week 5: Karpathy and Flowers

This week focused on transfer learning, model evaluation, and practical neural network training techniques. I read Andrej Karpathy's essay "A Recipe for Training Neural Networks" and reviewed relevant sections of Géron's Hands-On Machine Learning.

Using MobileNetV2 as a frozen feature extractor, I trained a flower classification model on the TensorFlow Flowers dataset containing five classes: Daisy, Dandelion, Roses, Sunflowers, and Tulips.

The model achieved a validation accuracy of 89.51%, exceeding the target accuracy of 88%.

Confusion matrix analysis showed that most errors occurred between visually similar flower categories, highlighting the importance of examining misclassified examples rather than relying solely on accuracy metrics.

This project reinforced the value of data inspection, transfer learning, visualization, and systematic model debugging.


# Week 6: TrashNet Exploratory Data Analysis (EDA)

## Overview

This week focused on performing Exploratory Data Analysis (EDA) on the TrashNet dataset to understand its structure and quality before training an image classification model.

## Tasks Completed

* Downloaded and organized the TrashNet dataset into the project directory.
* Counted the number of images available in each waste category.
* Visualized random sample images from all six classes.
* Analyzed image dimensions and file size distributions.
* Checked the dataset for corrupt images using `PIL.Image.verify()`.
* Created a reproducible stratified train-validation-test split (70%/15%/15%).
* Saved the train, validation, and test file lists as CSV files.
* Documented observations about class imbalance, image variability, and similarities between different waste categories.

## Dataset Classes

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

## Files Added

* `notebooks/w6_trashnet_eda.ipynb`
* `notebooks/train.csv`
* `notebooks/val.csv`
* `notebooks/test.csv`

## Outcome

Successfully completed the exploratory data analysis of the TrashNet dataset, providing a strong foundation for developing and evaluating waste image classification models in the upcoming weeks.


