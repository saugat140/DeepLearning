# Deep Learning Portfolio

Selected projects from my Deep Learning coursework, built with PyTorch (plus Keras and Ultralytics YOLOv8) and run on Google Colab GPUs. Each folder has its own README with the problem, approach, results and how to run it.

| Project | Task | Models | Headline result |
|---|---|---|---|
| [Crop & Weed Detection](crop-weed-detection-yolov8/) | Object detection + per-image plant counting | YOLOv8m (fine-tuned from pre-trained weights) | mAP50 0.878, mAP50-95 0.576, precision 0.810, recall 0.799 on 130 validation images; adds per-image crop/weed counting |
| [Retinal Vessel Segmentation](retinal-vessel-segmentation-unet/) | Semantic segmentation of fundus images | U-Net with 2, 3 and 4 encoder blocks (from scratch) | Best model (3 blocks): mean Dice 0.795, IoU 0.661 on 20 held-out images |
| [ENZYMES Graph Classification](graph-classification-enzymes-gcn/) | 6-class graph classification | 1- and 2-layer GCN (PyTorch Geometric) | 2-layer GCN: macro-F1 0.536, macro AUC 0.840 on 60 test graphs |
| [MNIST Architecture Comparison](mnist-architecture-comparison/) | 10-class image classification | DNN, custom CNN, ResNet18 (from scratch) | ResNet18 at lr 0.001: 98.8% test accuracy, macro-F1 0.988 |
| [Cancer Mortality Regression](cancer-mortality-regression/) | Tabular regression | Linear regression vs. four Keras DNNs | Baseline vs. depth comparison with train/val loss curves |

## Tech stack

Python, PyTorch, torchvision, PyTorch Geometric, TensorFlow/Keras, Ultralytics YOLOv8, scikit-learn, OpenCV, NumPy, Pandas, Matplotlib, Google Colab (CUDA).

## What these projects cover

- Data preparation: train/validation/test splits (stratified where it matters), normalization, label and file-consistency checks
- Training workflows: Adam/SGD, learning-rate scheduling, validation-based checkpointing, loss functions (cross-entropy, Dice + BCE)
- Evaluation: accuracy, macro-F1, ROC/AUC, confusion matrices, Dice/IoU, mAP, precision/recall
- Controlled comparisons: depth, learning rate and normalization ablations

## Notes on scope

These are course projects trained on single-GPU Colab runtimes, not production systems. Some test sets are small (20 images for segmentation, 60 graphs for ENZYMES), so the numbers are indicative rather than definitive. I have not covered model deployment/serving or multi-GPU training in this repo.

## Running the notebooks

Open any notebook in Google Colab and select a GPU runtime. Dependencies are installed in the first cells (`pip install ...`), and datasets are downloaded inside the notebooks (Kaggle for the crop/weed data, torchvision for MNIST, PyTorch Geometric for ENZYMES). The retinal segmentation notebook expects the course-provided `Data.zip`, which is not included.
