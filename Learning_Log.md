# Learning Log

## Week 3

This week was the first time I clearly saw the difference between a model that memorizes training data and a model that generalizes well. The training accuracy of the baseline CNN became very high, but the validation accuracy stopped improving. This helped me understand overfitting in a practical way rather than just reading about it.

The most interesting thing I learned was data augmentation. Before this week, I thought improving performance always meant changing the architecture or adding more layers. I learned that simply creating more diverse training examples can significantly improve performance.

One thing that still confuses me is determining exactly which augmentation techniques are most beneficial for a specific dataset. I understand that horizontal flipping works for CIFAR-10, but I am not yet fully confident about when rotations, zooming, or other transformations should be used.

Overall, this week helped me understand one of the most important challenges in deep learning: balancing model capacity with generalization.


Chosen Backbone: EfficientNetB0

I selected EfficientNetB0 as the primary backbone for the remainder of the internship. It provided strong validation accuracy while maintaining a relatively small model size and reasonable training speed. This makes it a practical choice for experimentation on Colab's free GPU and future deployment projects.


## Week 4 Reflection

This week introduced transfer learning and fine-tuning. The most surprising observation was that fine-tuning improved validation accuracy even though only a small portion of the network weights were updated. I expected a larger gap between the three backbone architectures, but the results were relatively close.

Another important lesson was the role of preprocessing functions. Each backbone required its own preprocess_input function, and using the wrong preprocessing could silently reduce model performance without producing any error message.

I also experienced several environment issues involving TensorFlow, TensorFlow Datasets, and package version conflicts. Troubleshooting these problems helped me understand how machine learning environments and dependencies affect reproducibility.

### Week 5 Learning Log

Today I worked with the Flowers dataset and applied Andrej Karpathy's overfit-one-batch sanity check. I trained an EfficientNetB0 transfer learning model on two images and successfully reduced the loss from 1.3561 to 0.0035 while achieving 100% accuracy. This verified that the training pipeline was functioning correctly.

After the sanity check passed, I trained the model on the full Flowers dataset and achieved 100% validation accuracy after two epochs. This exercise helped me understand how to debug machine learning pipelines before running full-scale training experiments.


# LEARNING LOG

## Week 5 - Karpathy and Flowers

### What I Read

* Andrej Karpathy: A Recipe for Training Neural Networks
* Relevant sections from Hands-On Machine Learning by Aurélien Géron

### Key Ideas Learned

1. Become one with the data.

   * Always inspect and understand the dataset before training.

2. Start with a simple baseline.

   * Build a working model first, then improve it incrementally.

3. Overfit a small batch first.

   * Verify that the training pipeline works correctly before large-scale training.

4. Visualize everything.

   * Training curves and confusion matrices help identify problems.

5. Inspect misclassified examples.

   * Incorrect predictions often reveal the true weaknesses of a model.

### Application to Flowers Project

* Used MobileNetV2 for transfer learning.
* Achieved 89.51% validation accuracy.
* Generated and analyzed a confusion matrix.
* Observed confusion between visually similar flower classes.
* Learned the importance of evaluating model errors rather than relying only on accuracy.

* # Week 8 – Hyperparameter Sweep Reflection

This week focused on understanding how different hyperparameters influence the performance of a deep learning model rather than simply searching for the highest accuracy.

During previous weeks, I changed parameters such as image size, dropout rate, learning rate, augmentation, and the number of unfrozen layers without systematically comparing their effects. This week, I followed a structured experimental approach.

I selected three important hyperparameters for the sweep:

- Image Size
- Dropout Rate
- Data Augmentation Strength

Instead of testing every possible combination, I used a six-run fractional design to reduce computation while still exploring different regions of the hyperparameter space.

One important lesson I learned was that changing multiple parameters simultaneously makes it difficult to understand why performance changes. Keeping the rest of the training pipeline constant allowed me to compare each experiment fairly.

I also learned the importance of recording every experiment in a structured format using experiments.csv. Maintaining proper experiment records makes future comparisons much easier and improves reproducibility.

Another key takeaway was that higher complexity does not always produce better performance. Larger image sizes, stronger augmentation, or higher dropout values are not guaranteed to improve accuracy. The best configuration is often a balance between accuracy, training time, and model stability.

Overall, this week improved my understanding of experimental design, hyperparameter tuning, and systematic model evaluation. These are essential skills for developing reliable deep learning models.


