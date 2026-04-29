import kagglehub

# Download latest version
path = kagglehub.dataset_download("jayjoshi37/reelsshorts-consumption-vs-attention-span")

print("Path to dataset files:", path)

"""
COMP 2200 — Intro to Machine Learning
Assignment: Noisy Sensors, Two Models, One Pipeline (20 points)

SUBMIT: .ipynb or .py
ALLOWED: numpy, matplotlib
NOT ALLOWED (for training): scikit-learn, statsmodels, tensorflow/torch

Your job:
1) Generate noisy sensor data (regression target) and risk labels (classification).
2) Split train/test.
3) Implement:
   - Linear regression with L2 regularization (Ridge-style)
   - Logistic regression with L2 regularization
4) Evaluate with MSE + classification metrics.
5) Produce diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionL1:
    def __init__(self, lr=0.01, lmbda=0.1, epochs=1000):
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z)) # sigmoid (0, 1)
    
    def fitL2(self, x, y):
        n = len(x) # num of samples

        # implement our training loop
        for _ in range(self.epochs):
            # 1) Linear score
            z = self.w * x + self.b # normal regression so far...

            # 2) Probability via sigmoid
            # y_pred = sigmoid(z)
            y_pred = self.sigmoid(z) # make it binary

            # 3) Gradients with L2 penalty on w

            # gradient of BCE + L2 on slope
            dw = (1/n) * np.sum((y_pred - y) * x) + (self.lmbda * self.w)
            db = (1/n) * np.sum(y_pred - y)


            # 4) Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def fitL1(self, x, y):
        n = len(x)
        for _ in range(self.epochs):
            z = self.w * x + self.b
            y_pred = self.sigmoid(z)


            
            


    def predict(self, X_test):
        return self.w * X_test + self.b




# -------------------------------------------------------------------
# PART 0: Reproducibility
# -------------------------------------------------------------------
np.random.seed(42)


# -------------------------------------------------------------------
# PART 1: Noisy Data Generation
# -------------------------------------------------------------------
def generate_noisy_data():
    """
    Generates time t, noisy volume v_noisy, and class labels y_class.

    Underlying clean physics:
        V(t) = ((t - 4)^3)/64 + 3.3

    Noise:
        Gaussian noise with std = 0.15 is added to volume.

    Labels:
        y_class = 1 if v_clean > 4.5 else 0
        Note: labels come from v_clean (NOT v_noisy), creating a fuzzy boundary.
    """
    t = np.linspace(0, 10, 300)
    v_clean = ((t - 4)**3 / 64) + 3.3

    noise = np.random.normal(0, 0.15, size=t.shape)
    v_noisy = v_clean + noise

    y_class = (v_clean > 4.5).astype(int)
    return t, v_noisy, y_class


# -------------------------------------------------------------------
# PART 2: Train-Test Split
# -------------------------------------------------------------------
def train_test_split(x, y1, y2, test_size=0.2):
    """
    Shuffles and splits arrays into train/test.

    Returns:
        x_train, x_test, y1_train, y1_test, y2_train, y2_test
    """
    indices = np.random.permutation(len(x))
    test_set_size = int(len(x) * test_size)

    test_idx = indices[:test_set_size]
    train_idx = indices[test_set_size:]

    return (x[train_idx], x[test_idx],
            y1[train_idx], y1[test_idx],
            y2[train_idx], y2[test_idx])


# -------------------------------------------------------------------
# PART 4: Logistic Regression with L2 Regularization
# -------------------------------------------------------------------


def train_logistic_regression_l2(x, y, lr=0.1, lmbda=0.1, epochs=2000):
    """
    Train logistic regression p = sigmoid(w*x + b) using gradient descent.

    IMPORTANT: Include L2 penalty on w (NOT on b).

    Gradient (one common convention):
        dw = (1/n)*sum((y_pred - y)*x) + (lmbda*w)
        db = (1/n)*sum(y_pred - y)

    Implement gradient descent updates for w and b.

    Return:
        w, b
    """
    w, b = 0.0, 0.0
    n = len(x)

    for _ in range(epochs):
        # 1) Linear score
        # z = ...
        z = w * x + b # normal regression so far...

        # 2) Probability via sigmoid
        # y_pred = sigmoid(z)
        y_pred = sigmoid(z) # make it binary

        # 3) Gradients with L2 penalty on w
        # dw = ...
        # db = ...

        # gradient of BCE + L2 on slope
        dw = (1/n) * np.sum((y_pred - y) * x) + (lmbda * w)
        db = (1/n) * np.sum(y_pred - y)


        # 4) Update parameters
        # w = ...
        # b = ...
        w -= lr * dw
        b -= lr * db

    return w, b


# -------------------------------------------------------------------
# PART 5: Evaluation Metrics
# -------------------------------------------------------------------
def get_metrics(y_true, y_prob):
    """
    Computes accuracy, precision, recall, f1 using threshold 0.5.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, (tp, tn, fp, fn)


# -------------------------------------------------------------------
# EXECUTION PIPELINE (Students fill in TODOs above)
# -------------------------------------------------------------------

# A) Generate data
t, v, y = generate_noisy_data()

# Required Plot (Part A)
plt.figure(figsize=(8, 4))
plt.scatter(t, v, alpha=0.6)
plt.title("Noisy Sensor Volume vs Time")
plt.xlabel("Time (hours)")
plt.ylabel("Noisy Volume")
plt.grid(True, alpha=0.3)
plt.show()

print("\nPART A (Write-up prompt):")
print("In 2–4 sentences, explain why labels y are computed from v_clean instead of v_noisy,")
print("and what a 'fuzzy boundary' means here.\n")

# B) Split
t_train, t_test, v_train, v_test, y_train, y_test = train_test_split(t, v, y, test_size=0.2)

print("Train/Test sizes:")
print("t_train:", t_train.shape, "t_test:", t_test.shape)
print("v_train:", v_train.shape, "v_test:", v_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

print("\nPART B (Write-up prompt):")
print("Confirm your split is about 80/20. (Show the counts or percentages.)\n")


# C) Linear regression with L2
# You may tune lmbda (regularization strength) as part of your analysis.
m, b_lin = train_linear_regression_l2(t_train, v_train, lr=0.01, lmbda=0.05, epochs=1000)

# Predictions + MSE
v_pred = m * t_test + b_lin
mse = np.mean((v_test - v_pred)**2)

print("\n--- Linear Regression Results ---")
print("m (slope):", m)
print("b (intercept):", b_lin)
print(f"Test MSE: {mse:.4f}")

print("\nPART C (Write-up prompt):")
print("In 3–6 sentences: What happens to the learned slope when lmbda increases?")
print("Why does that make sense when the data is noisy?\n")

# D) Logistic regression with L2
w, b_log = train_logistic_regression_l2(t_train, y_train, lr=0.1, lmbda=0.2, epochs=2000)
test_probs = sigmoid(w * t_test + b_log)

acc, prec, rec, f1, counts = get_metrics(y_test, test_probs)
tp, tn, fp, fn = counts

print("\n--- Logistic Regression Results ---")
print("w (weight):", w)
print("b (bias):", b_log)
print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nPART D (Write-up prompt):")
print("In 2–3 sentences: Why do we clip z before np.exp in sigmoid?")
print("What problem does it prevent?\n")
# -------------------------------------------------------------------
# PART E: Metrics Visualization Block (Required)
# -------------------------------------------------------------------

# 1) Residuals (Linear Regression)
residuals = v_test - v_pred

# 2) Confusion Matrix counts (Logistic Regression)
y_pred_class = (test_probs >= 0.5).astype(int)
tp = np.sum((y_test == 1) & (y_pred_class == 1))
tn = np.sum((y_test == 0) & (y_pred_class == 0))
fp = np.sum((y_test == 0) & (y_pred_class == 1))
fn = np.sum((y_test == 1) & (y_pred_class == 0))

# 3) Plot grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3)

# Plot A: Residual scatter
axes[0, 0].scatter(v_pred, residuals, alpha=0.5)
axes[0, 0].axhline(0, linestyle='--')
axes[0, 0].set_title("Residual Plot (Linear Regression)")
axes[0, 0].set_xlabel("Predicted Volume")
axes[0, 0].set_ylabel("Residuals (Actual - Predicted)")
axes[0, 0].grid(True, alpha=0.3)

# Plot B: Residual histogram
axes[0, 1].hist(residuals, bins=15, edgecolor="black", alpha=0.7)
axes[0, 1].set_title("Distribution of Residuals")
axes[0, 1].set_xlabel("Residual Value")
axes[0, 1].set_ylabel("Frequency")

# Plot C: Confusion Matrix
cm = np.array([[tn, fp], [fn, tp]])
im = axes[1, 0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["No Risk (0)", "Risk (1)"])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(["No Risk (0)", "Risk (1)"])

for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")

# Plot D: Metrics bar chart
metrics_vals = [acc, prec, rec, f1]
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
axes[1, 1].bar(metric_names, metrics_vals, alpha=0.8)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_title("Classification Metrics")

for i, val in enumerate(metrics_vals):
    axes[1, 1].text(i, val + 0.02, f"{val:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()