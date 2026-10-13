# Crop & Weed Detection with YOLOv8

Detects crops and weeds in field images and counts each class per image, using a YOLOv8m model fine-tuned from pre-trained weights.

**Notebook:** [`crop_weed_detection_yolov8.ipynb`](crop_weed_detection_yolov8.ipynb)

## Problem

Given field photos, localize and classify every plant as `crop` or `weed` (2 classes), so weed density can be estimated per image.

## Data

[Kaggle: crop and weed detection data with bounding boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes), downloaded in the notebook with `kagglehub`. Images and YOLO-format labels are reorganized into `images/{train,val}` and `labels/{train,val}` with a random 90/10 split. The notebook also checks that every image has a matching label file and prints class counts for the training labels.

## Approach

- Model: YOLOv8m (Ultralytics), initialized from pre-trained `yolov8m.pt` weights and fine-tuned
- Training: up to 40 epochs, batch size 8, 640 px images, early stopping (patience 10)
- Evaluation: mAP50, mAP50-95, precision and recall on the validation split, plus loss/mAP/precision/recall curves from `results.csv`
- Inference: predictions at confidence 0.5 with saved sample outputs
- Counting: `detect_and_count()` returns per-class counts for an image (e.g. `{crop: n, weed: m}`)
- Efficiency: reports weights size (MB) and average inference time / FPS over 50 images

## Results

<!-- TODO: paste mAP50, mAP50-95, precision, recall, weights size and FPS from the notebook output, and add 2-3 sample prediction images to an `images/` folder. -->

See the notebook outputs for the metrics table and training curves.

## Limitations

The validation split doubles as the test split, and inference demos use images from the training folder, so the numbers describe in-distribution performance on this dataset only.

## Run it

Open the notebook in Google Colab with a GPU runtime. The first cell installs `ultralytics kagglehub opencv-python pyyaml matplotlib seaborn pandas`.