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
