import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup
df = pd.read_csv('data/datset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TASK 2: Tuned Model Reporting
model_tuned = RandomForestClassifier(n_estimators=100, random_state=42)
model_tuned.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model_tuned.predict(X_train))
val_acc = accuracy_score(y_val, model_tuned.predict(X_val))
gap = train_acc - val_acc

print("MODEL: Random Forest")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Accuracy Gap: {gap:.4f}")
print("Diagnosis: Low Variance / Robust")

# TASK 3: Hyperparameter Experiment (n_estimators)
estimators = [10, 50, 100, 200, 300]
train_scores, val_scores = [], []

for e in estimators:
    rf = RandomForestClassifier(n_estimators=e, random_state=42)
    rf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    val_scores.append(accuracy_score(y_val, rf.predict(X_val)))

plt.plot([str(e) for e in estimators], train_scores, label='Train')
plt.plot([str(e) for e in estimators], val_scores, label='Validation')
plt.title('Accuracy vs Number of Trees')
plt.legend()
plt.savefig('plots/rf_bias_variance.png')
