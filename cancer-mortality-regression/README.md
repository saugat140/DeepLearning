# Cancer Mortality Regression: Linear Regression vs. DNNs

Predicts county-level cancer mortality rate (`TARGET_deathRate`) from demographic and socioeconomic features, comparing a linear regression baseline against fully connected networks of increasing depth.

**Script:** [`Linear Regression and DNN Models.py`](Linear%20Regression%20and%20DNN%20Models.py) · **Data:** [`cancer_reg-1.csv`](cancer_reg-1.csv)

## Approach

- Data: rows with missing values dropped; `Geography` and `binnedInc` removed; features standardized with `StandardScaler` fit on the training split only
- Split: 70% train / 15% validation / 15% test (random_state 42)
- Baseline: scikit-learn `LinearRegression`
- DNNs (Keras, ReLU hidden layers, linear output, SGD, MSE loss, 50 epochs, batch size 32): `[16]`, `[30, 8]`, `[30, 16, 8]` and `[30, 16, 8, 4]`
- Evaluation: test MSE and R² for every model, plus train/validation loss curves for the DNNs

<!-- TODO: run the script once and paste the printed "Model Results (Test Set)" table (MSE and R² per model) here, and save the loss-curve plot to an `images/` folder. -->

## Run it

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
python "Linear Regression and DNN Models.py"
```

The CSV must sit in the same folder as the script.
