# MNIST: DNN vs. CNN vs. ResNet18

Compares three architectures on MNIST digit classification across three learning rates, all trained from scratch in PyTorch.

**Notebook:** [`mnist_architecture_comparison.ipynb`](mnist_architecture_comparison.ipynb)

## Approach

- Data: MNIST normalized with mean 0.1307 / std 0.3081; 48,000 train / 12,000 validation (random split of the training set) / 10,000 test
- Models: a fully connected DNN (784-128-64-10), a custom 2-conv-layer CNN, and ResNet18 with its first convolution adapted to 1-channel input
- Training: cross-entropy loss, Adam, 5 epochs, batch size 64, learning rates 0.1 / 0.01 / 0.001; the best epoch by validation macro-F1 is kept
- Evaluation: test accuracy and macro-F1 for each model and learning rate

## Results (10,000 test images)

| Model | Learning rate | Test accuracy | Test macro-F1 |
|---|---|---|---|
| DNN | 0.1 | 0.1150 | 0.0233 |
| DNN | 0.01 | 0.9524 | 0.9521 |
| DNN | 0.001 | 0.9654 | 0.9652 |
| CNN | 0.1 | 0.1135 | 0.0204 |
| CNN | 0.01 | 0.9776 | 0.9774 |
| CNN | 0.001 | 0.9840 | 0.9839 |
| ResNet18 | 0.1 | 0.9827 | 0.9825 |
| ResNet18 | 0.01 | 0.9823 | 0.9823 |
| ResNet18 | 0.001 | **0.9877** | **0.9876** |

## Takeaways

- Learning rate 0.1 made the DNN and CNN fail to train (about 11% accuracy, i.e. chance), while ResNet18 stayed stable at every learning rate, likely thanks to its BatchNorm layers.
- Accuracy improved with architecture capacity at the best learning rate: DNN 96.5% < CNN 98.4% < ResNet18 98.8%.
- Single run per configuration, so small gaps (e.g. CNN vs. ResNet18) should not be over-interpreted.

## Run it

Open the notebook in Google Colab; MNIST downloads automatically through torchvision.
