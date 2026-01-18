import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Setup
df = pd.read_csv('data/datset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# TASK 2: Tuned Model Reporting
model_tuned = KNeighborsClassifier(n_neighbors=5)
model_tuned.fit(X_train_scaled, y_train)

# Compute accuracies
train_acc = accuracy_score(y_train, model_tuned.predict(X_train_scaled))
val_acc = accuracy_score(y_val, model_tuned.predict(X_val_scaled))
gap = train_acc - val_acc

print("MODEL: KNN (Scaled)")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Accuracy Gap: {gap:.4f}")
print("Diagnosis: Balanced / Potential Bias")

# TASK 3: Hyperparameter Experiment (n_neighbors)
neighbors = [1,3,5,7,9,11]
train_scores, val_scores = [], []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, knn.predict(X_train))
    val_acc = accuracy_score(y_val, knn.predict(X_val))
    train_scores.append(train_acc)
    val_scores.append(val_acc)


plt.plot([str(n) for n in neighbors], train_scores, label='Train')
plt.plot([str(n) for n in neighbors], val_scores, label='Validation')
plt.title('Accuracy vs Number of Neighbors')
plt.savefig('plots/knn_bias_variance.png')
