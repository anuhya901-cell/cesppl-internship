# Learning Log

## Week 3

This week was the first time I clearly saw the difference between a model that memorizes training data and a model that generalizes well. The training accuracy of the baseline CNN became very high, but the validation accuracy stopped improving. This helped me understand overfitting in a practical way rather than just reading about it.

The most interesting thing I learned was data augmentation. Before this week, I thought improving performance always meant changing the architecture or adding more layers. I learned that simply creating more diverse training examples can significantly improve performance.

One thing that still confuses me is determining exactly which augmentation techniques are most beneficial for a specific dataset. I understand that horizontal flipping works for CIFAR-10, but I am not yet fully confident about when rotations, zooming, or other transformations should be used.

Overall, this week helped me understand one of the most important challenges in deep learning: balancing model capacity with generalization.
