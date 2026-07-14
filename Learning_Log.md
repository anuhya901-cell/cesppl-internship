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

# End of Rehearsal

The first eight weeks of this internship were a rehearsal for working with the real CESPPL operational dataset. At the start, I could train basic models, but I did not yet have a reliable process for diagnosing failures, comparing experiments, or understanding why a model made mistakes.

The TrashNet work changed that. I learned that model development is not only about calling `model.fit()` and checking accuracy. A complete workflow includes validating the dataset, checking labels, establishing a baseline, reading training curves, evaluating every class, studying errors, and recording experiments carefully.

One major lesson was that more complicated techniques do not automatically improve results. Fine-tuning performed worse than feature extraction in some runs. DenseNet121 did not outperform EfficientNetB0 in my implementation. Class weighting slightly reduced overall accuracy but improved recall for the rare Trash class. The hyperparameter sweep showed that image size had a clearer effect than some of the other settings, while strong augmentation was not always helpful.

I also learned to treat errors as useful information. Looking at confusion matrices and misclassified images helped me understand that several TrashNet mistakes were caused by visual similarity, poor image quality, and ambiguous labels rather than complete model failure.

## Five Concrete Skills I Can Now Perform

1. **Build a complete transfer-learning pipeline**

   I can load an image dataset, create train and validation splits, apply data augmentation, build an EfficientNetB0 or DenseNet121 classifier, compile it, train it with callbacks, and save the model and weights.

2. **Perform controlled fine-tuning**

   I can unfreeze a chosen number of backbone layers, recompile with a smaller learning rate, keep the base model in inference mode, and compare fine-tuning fairly against the feature-extraction baseline.

3. **Evaluate models beyond accuracy**

   I can generate and interpret confusion matrices, classification reports, per-class precision, recall, and F1-score. I can also identify which classes are being confused most often.

4. **Perform systematic error analysis**

   I can collect misclassified examples, display them in readable grids, inspect prediction confidence, categorize likely causes of error, and use Grad-CAM on representative samples.

5. **Design and track experiments**

   I can run a fractional hyperparameter sweep, change only selected parameters, record results in `experiments.csv`, compare accuracy against training time, select a winner, and document the reasoning clearly.

## Reflection on My Process

During the earlier weeks, I sometimes moved too quickly and skipped checks that should have happened before training. I learned that verifying the dataset split and class labels is essential. One evaluation mistake produced results for only two classes, which showed me why checking unique labels before training and evaluation is necessary.

I also learned that I must read training curves carefully instead of relying only on the final printed accuracy. Early stopping may restore a better checkpoint than the last epoch, and validation loss often gives an earlier warning about overfitting.

The error-analysis work required patience. Looking at multiple misclassified images was slower than running another model, but it gave more useful understanding of the data. In future work, I should not skip this step.

The iteration playbook is one of the most useful outputs from these eight weeks. It gives me a repeatable process for handling low accuracy or unstable training without making random changes.

From the next week onward, I will apply this workflow to the real CESPPL operational data. My priority will be data quality, reproducibility, careful evaluation, and clear documentation rather than chasing accuracy without understanding the result.


