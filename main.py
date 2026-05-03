import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) DATA LOADING & PREPROCESSING
# -------------------------------------------------------------------

filename = "reels_attention_span_dataset_12000.csv"
df = pd.read_csv(filename)

# --- FEATURE SELECTION ---
# We drop 'user_id' because it's just a label, not a real data point.
df = df.drop('user_id', axis=1)

# --- HANDLING CATEGORICAL DATA (The 'platform' column) ---
# One-Hot Encoding: This creates new columns for 'Instagram Reels' and 'YouTube Shorts' 
# containing 1s and 0s so the model can understand the platform.
df = pd.get_dummies(df, columns=['platform'], drop_first=True, dtype=int)

# --- DEFINING TARGET (y) ---
# We want to predict Stress Level. 
# We turn it into 1 if 'High', and 0 if 'Medium' or 'Low'.
y = (df['stress_level'] == 'High').astype(int).values

# --- DEFINING FEATURES (X) ---
# Use all columns EXCEPT the target 'stress_level'
X_raw = df.drop('stress_level', axis=1).values.astype(float)

# DATA SCALING: (X - mean) / std
# This is critical for L1/L2 regularization to work fairly across all features.
X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)


# TRAIN/TEST SPLIT (80/20)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_point = int(len(X) * 0.8)
train_idx, test_idx = indices[:split_point], indices[split_point:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# -------------------------------------------------------------------
# 2) THE MODEL CLASS (BASELINE + EXTENSION)
# -------------------------------------------------------------------

class LogisticRegression:
    def __init__(self, lr=0.01, lmbda=0.1, epochs=1500):
        """
        lr: Learning Rate (size of the step we take during training)
        lmbda: Regularization strength (how much we punish large weights)
        epochs: How many times we look at the whole dataset
        """
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        """Standard math function to squash any number into a range between 0 and 1."""
        # np.clip prevents math errors if z is a giant number
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, penalty='L2'):
        """
        This is the training loop.
        penalty='L2' is our BASELINE (Ridge).
        penalty='L1' is our EXTENSION (Lasso).
        """
        n_samples, n_features = X.shape
        # We start with all weights at zero
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            # STEP 1: Linear Math (z = X*w + b)
            # np.dot handles all features/columns at once
            z = np.dot(X, self.w) + self.b

            # STEP 2: Turn the math into a probability (0 to 1)
            y_pred = self.sigmoid(z)

            # STEP 3: Calculate the error (Difference between guess and truth)
            error = y_pred - y

            # STEP 4: Calculate Gradient (Direction to move weights to fix the error)
            # Standard gradient for logistic regression
            dw = (1 / n_samples) * np.dot(X.T, error)

            # APPLY REGULARIZATION (The core of our project)
            if penalty == 'L2':
                # BASELINE: Gradient of squared weights is just the weight itself
                dw += (self.lmbda * self.w)
            elif penalty == 'L1':
                # EXTENSION: Gradient of absolute weights is the 'sign' (1 or -1)
                dw += (self.lmbda * np.sign(self.w))
            
            db = (1/n_samples) * np.sum(error)

            # STEP 5: Update parameters (Gradient Descent)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X_test):
        """Uses the learned weights to predict 0 or 1 for new data."""
        z = np.dot(X_test, self.w) + self.b
        probabilies = self.sigmoid(z)
        # If probability is 0.5 or higher, we predict class 1
        return (probabilies >= 0.3).astype(int)
    
# -------------------------------------------------------------------
# 3) EXECUTION AND EVALUATION
# -------------------------------------------------------------------

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return 2 * (precision * recall) / (precision + recall + 1e-8)

# initialize one model instance for each regularization type
model = LogisticRegression(lr=0.1, lmbda=0.2)

# train the baseline (L2)
model.fit(X_train, y_train, penalty='L2')
base_preds = model.predict(X_test)
base_accuracy = np.mean(base_preds == y_test)
base_f1 = f1_score(y_test, base_preds)
weights_baseline = model.w.copy() # save weights for the graph

# train the extension (L1)
model = LogisticRegression(lr=0.1, lmbda=0.5)
model.fit(X_train, y_train, penalty='L1')
ext_preds = model.predict(X_test)
ext_accuracy = np.mean(ext_preds == y_test)
ext_f1 = f1_score(y_test, ext_preds)
weights_extension = model.w.copy() # save weights for the graph

print(f'L2 Accuracy: {base_accuracy:.2%}, F1: {base_f1:.4f}')
print(f'L1 Accuracy: {ext_accuracy:.2%}, F1: {ext_f1:.4f}')

print("Predicted positives (L2):", np.sum(base_preds))
print("Predicted positives (L1):", np.sum(ext_preds))
print("Actual positives:", np.sum(y_test))

print("L2 near-zero weights:", np.sum(np.abs(weights_baseline) < 1e-3))
print("L1 near-zero weights:", np.sum(np.abs(weights_extension) < 1e-3))

# -------------------------------------------------------------------
# 4) PLOTS
# -------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PLOT 1: Performance Bar Chart
# Shows which model actually predicted better
ax1.bar(['L2 (Baseline)', 'L1 (Extension)'], [base_accuracy, ext_accuracy], color=['skyblue', 'salmon'])
ax1.set_title('Accuracy Comparison')
ax1.set_ylabel('Accuracy %')
ax1.set_ylim(min(base_accuracy, ext_accuracy) - 0.02,
             max(base_accuracy, ext_accuracy) + 0.02)
ax1.set_yticks(np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 5))
ax1.set_yticklabels([f"{y:.2%}" for y in ax1.get_yticks()])

# PLOT 2: Weight Sparsity Chart
# This proves the L1 extension worked! L1 pushes unimportant weights to 0.
ax2.plot(weights_baseline, 'o-', label='Baseline (L2) Weights', alpha=0.7)
ax2.plot(weights_extension, 'x--', label='Extension (L1) Weights', alpha=0.9)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Impact on Feature Weights')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Weight Value')
ax2.legend()

plt.tight_layout()
plt.show()