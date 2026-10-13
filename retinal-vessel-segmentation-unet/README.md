# Retinal Vessel Segmentation with U-Net

Segments blood vessels in 512×512 retinal (fundus) images with U-Nets of different depth, trained from scratch in PyTorch.

**Notebook:** [`retinal_vessel_segmentation_unet.ipynb`](retinal_vessel_segmentation_unet.ipynb)

## Problem

Binary segmentation: classify every pixel as vessel or background.

## Data

100 fundus images with binary masks (80 train / 20 test), all 512×512, with no missing image/mask pairs. Images are scaled to [0, 1] and masks are binarized at a threshold of 127. The dataset was provided with the course and is not included in this repo.

## Approach

- Custom U-Net implemented from scratch with 2, 3 and 4 encoder/decoder blocks (base width 32, skip connections, transposed-conv upsampling, sigmoid output)
- Loss: BCE + Dice loss; optimizer: Adam (lr 1e-3); 15 epochs; batch size 4
- Metrics: mean Dice and IoU on the held-out test images, computed on both normalized and raw masks (identical results)
- Visual check: input / ground truth / prediction panels for sample images

## Results (20 held-out test images)

| U-Net depth | Mean Dice | Mean IoU |
|---|---|---|
| 2 blocks | 0.788 | 0.651 |
| 3 blocks | **0.795** | **0.661** |
| 4 blocks | 0.787 | 0.651 |

## Takeaways and limitations

- The 3-block model was best, but the gaps between depths are small and the test set is only 20 images, so the ranking is not statistically strong.
- No separate validation set was used, and results come from a single run per model.

## Run it

Open the notebook in Google Colab with a GPU runtime and upload the course `Data.zip` when prompted (it is unzipped to `/content/HW3`).
