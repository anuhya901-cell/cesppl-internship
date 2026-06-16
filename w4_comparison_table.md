| Backbone | Parameters | Feature Extraction Accuracy | Fine-Tuning Accuracy | Time/Epoch | Weight Size |
|-----------|-----------|-----------|-----------|-----------|-----------|
| MobileNetV2 | 2,257,984 | 98.36% | 98.2% | 30 sec | 14 MB |
| ResNet50V2 | 2,268,865 | 92.0% | 93.6% | 80 sec | 98 MB |
| EfficientNetB0 | 5,834,908 | 96.3% | 98.2% | 36 sec | 22 MB |


## Conclusion

MobileNetV2 is the most suitable model for deployment on a mobile phone because it has the smallest number of parameters, the smallest weight file size, and the fastest inference speed.

If unlimited computational resources are available, ResNet50V2 is a strong choice because of its larger capacity and ability to learn more complex features.

For the CESPPL project using Colab's free GPU, EfficientNetB0 provides the best balance between accuracy, model size, and training speed. It achieves high accuracy while requiring significantly less computation than ResNet50V2.
