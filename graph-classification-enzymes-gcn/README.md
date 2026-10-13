# ENZYMES Graph Classification with GCNs

Classifies protein graphs from the ENZYMES benchmark into 6 enzyme classes using Graph Convolutional Networks in PyTorch Geometric.

**Notebook:** [`enzymes_gcn_classification.ipynb`](enzymes_gcn_classification.ipynb)

## Problem

Graph-level classification: predict the enzyme class (6 classes) of each of 600 protein graphs, each node carrying 21 features.

## Approach

- Data: `TUDataset("ENZYMES", use_node_attr=True)`; node features standardized to zero mean and unit variance; 80/10/10 train/val/test split (480/60/60 graphs), stratified for the train split
- Models: a 1-layer GCN and a 2-layer GCN (hidden size 64, BatchNorm, ReLU, dropout 0.5 in the 2-layer model), global mean pooling, linear classifier
- Training: Adam (lr 1e-3), 200 epochs, `ReduceLROnPlateau` on validation macro-F1, best checkpoint saved by validation F1
- Evaluation: test accuracy and macro-F1, one-vs-rest ROC curves with macro AUC, confusion matrix

## Results (60 test graphs)

| Model | Accuracy | Macro-F1 | Macro AUC |
|---|---|---|---|
| GCN, 1 layer | 0.450 | 0.448 | n/a |
| GCN, 2 layers | **0.533** | **0.536** | **0.840** |

An earlier run without BatchNorm, feature normalization or the LR scheduler gave a 2-layer macro-F1 of 0.462 and AUC of 0.795, so those additions improved the 2-layer result.

## Limitations

The test set is only 60 graphs and results come from one seed (42), so differences between models are noisy. ENZYMES is a hard benchmark with limited data; the scores reflect that.

## Run it

Open the notebook in Google Colab with a GPU runtime. The first cell installs `torch-geometric`; the dataset downloads automatically.
