# cancer_mortality.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers



df = pd.read_csv("cancer_reg-1.csv", encoding="latin1")


print("Shape of dataset:", df.shape)
print("Missing values:\n", df.isnull().sum())

headers = list(df.columns.values)
print ("List of column headers: \n", headers)


df = df.dropna()

# Define features and label
X = df.drop(columns=["TARGET_deathRate","binnedInc","Geography"])
y = df["TARGET_deathRate"]


print("Basic stats:\n", df.describe())

# Compute only on numeric columns
numeric_df = df.select_dtypes(include=[np.number])

print("Minimum:", numeric_df.min().min())   # overall minimum value
print("Maximum:", numeric_df.max().max())   # overall maximum value




#print("Deathrate column \n",x)
#print ("TARGET_deathRate: ",y)

# Split into train, validation, test (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression -> MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")


# Step 3–4: Deep Neural Networks
# -----------------------------

def build_dnn(layers_config, input_dim):
    """Builds a fully connected DNN model from list of hidden layer sizes"""
    model = keras.Sequential()
    # Add hidden layers
    for units in layers_config:
        model.add(layers.Dense(units, activation="relu", input_dim=input_dim))
        input_dim = None  # only first layer needs input_dim
    # Output layer
    model.add(layers.Dense(1))  # regression output
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss="mse",
                  metrics=["mse"])
    return model

# Define DNN configs
dnn_configs = {
    "DNN-16": [16],
    "DNN-30-8": [30, 8],
    "DNN-30-16-8": [30, 16, 8],
    "DNN-30-16-8-4": [30, 16, 8, 4],
}

results = {}

for name, config in dnn_configs.items():
    print(f"\nTraining {name}...")
    model = build_dnn(config, input_dim=X_train.shape[1])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        verbose=0)  # set verbose=1 to see training

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)

    # Plot training loss curve
    plt.plot(history.history["loss"], label=f"{name} Train")
    plt.plot(history.history["val_loss"], label=f"{name} Val")

plt.title("Training Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# -----------------------------
# Step 6: Compare Results
# -----------------------------
print("\n=== Model Results (Test Set) ===")
print(f"Linear Regression: MSE={mse_lr:.4f}, R2={r2_lr:.4f}")
for name, (mse, r2) in results.items():
    print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")
