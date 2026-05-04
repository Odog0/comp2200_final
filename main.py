import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) DATA LOADING AND PREPROCESSING
# -------------------------------------------------------------------

DATA_FILE = "reels_attention_span_dataset_12000.csv"

# Load the Kaggle dataset.
df = pd.read_csv(DATA_FILE)

# Get rid of user_ids because its irrelevant for the model
df = df.drop("user_id", axis=1)

# Create binary classification target.
# Cutoff is from 6-10
y = (df["attention_span_score"] >= 6).astype(int).values

# Use all remaining columns except the target as input features.
X_df = df.drop("attention_span_score", axis=1)

# change categorical columns to numerical values
X_df = pd.get_dummies(
    X_df,
    columns=["platform", "stress_level"],
    drop_first=True,
    dtype=int
)

feature_names = X_df.columns.to_numpy()
X_raw = X_df.values.astype(float)

# Train/test split BEFORE SCALING BC DATA LEAKAGE = BAD = SAD AI :(
np.random.seed(42)
indices = np.random.permutation(len(X_raw))
split_point = int(len(X_raw) * (1 - .20))
train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train_raw = X_raw[train_idx]
X_test_raw = X_raw[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# Standardize using only the training set statistics.
mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)
std[std == 0] = 1

X_train = (X_train_raw - mean) / std
X_test = (X_test_raw - mean) / std


# -------------------------------------------------------------------
# 2) BASELINE AND EXTENDED MODEL
# -------------------------------------------------------------------

class LogisticRegression:
    """
    From-scratch logistic regression using NumPy.

    Baseline: penalty="L2"
    Extension: penalty="L1"
    """

    # lr is how big of a step the model takes when updating weights
    # lmbda is how much regularization changes weights
    # epochs is the number of training rounds
    def __init__(self, lr=0.1, lmbda=0.001, epochs=3000):
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        #Convert a linear score into a probability between 0 and 1.
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, penalty="L2"):
        # Train logistic regression with either L2 or L1 regularization.
        penalty = penalty.upper()
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            # Forward pass: compute probabilities.
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)

            # Logistic regression gradient.
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            if penalty == "L2":
                # Baseline: L2 regularization penalizes large weights.
                dw += (self.lmbda / n_samples) * self.w
                self.w -= self.lr * dw

            elif penalty == "L1":
                # Extension: L1 regularization encourages small weights to become zero.
                self.w -= self.lr * dw
                self.w = np.sign(self.w) * np.maximum(
                    0,
                    np.abs(self.w) - self.lr * self.lmbda
                )

            self.b -= self.lr * db

    def predict(self, X):
        # Predict class labels using a 0.5 probability threshold.
        # e.g >=0.5 = 1
        z = np.dot(X, self.w) + self.b
        probabilities = self.sigmoid(z)
        return (probabilities >= 0.5).astype(int)


# -------------------------------------------------------------------
# 3) TRAINING AND EVALUATION
# -------------------------------------------------------------------

def accuracy_score(y_true, y_pred):
    # comupte accuracy from scratch
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    # compute F1 from scratch
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8) # no divide by zero error
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


# Train baseline model with L2 regularization.
l2_model = LogisticRegression(lr=0.1, lmbda=0.001, epochs=3000)
l2_model.fit(X_train, y_train, penalty="L2")
l2_preds = l2_model.predict(X_test)
l2_accuracy = accuracy_score(y_test, l2_preds)
l2_f1 = f1_score(y_test, l2_preds)

# Train extended model with L1 regularization.
l1_model = LogisticRegression(lr=0.1, lmbda=0.001, epochs=3000) # note passing same lr, lmbda, and epochs for fair training
l1_model.fit(X_train, y_train, penalty="L1")
l1_preds = l1_model.predict(X_test)
l1_accuracy = accuracy_score(y_test, l1_preds)
l1_f1 = f1_score(y_test, l1_preds)

print("L2 Baseline")
print(f"Accuracy: {l2_accuracy:.2%}")
print(f"F1 Score: {l2_f1:.4f}")

print("\nL1 Extension")
print(f"Accuracy: {l1_accuracy:.2%}")
print(f"F1 Score: {l1_f1:.4f}")


# -------------------------------------------------------------------
# 4) VISUALIZATIONS
# -------------------------------------------------------------------

# compare baseline and extension performance.
model_names = ["L2 Baseline", "L1 Extension"]
accuracy_values = [l2_accuracy, l1_accuracy]
f1_values = [l2_f1, l1_f1]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, accuracy_values, width, label="Accuracy")
plt.bar(x + width / 2, f1_values, width, label="F1 Score")
plt.title("Baseline vs. Extended Model Performance")
plt.ylabel("Score")
plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")
plt.show()

# compare learned feature weights.
feature_index = np.arange(len(feature_names))

plt.figure(figsize=(10, 5))
plt.plot(feature_index, l2_model.w, "o-", label="L2 Baseline")
plt.plot(feature_index, l1_model.w, "x--", label="L1 Extension")
plt.axhline(0, linewidth=0.8)
plt.title("Learned Feature Weights")
plt.xlabel("Feature")
plt.ylabel("Weight Value")
plt.xticks(feature_index, feature_names, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("feature_weights.png", dpi=300, bbox_inches="tight")
plt.show()
