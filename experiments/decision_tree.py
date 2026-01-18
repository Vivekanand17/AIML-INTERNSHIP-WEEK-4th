import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Setup Data
df = pd.read_csv('../data/datset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TASK 3 Experiment: Varying max_depth
depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
train_scores = []
val_scores = []

print("Depth | Train Acc | Val Acc | Gap")
print("-" * 35)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    val_acc = accuracy_score(y_val, dt.predict(X_val))
    
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    
    depth_label = "None" if d is None else d
    print(f"{depth_label:<5} | {train_acc:.4f}    | {val_acc:.4f} | {train_acc-val_acc:.4f}")

# 3. Generate Mandatory Plot
plt.figure(figsize=(10, 6))
# Convert None to a string or number for plotting
plot_depths = [d if d is not None else 25 for d in depths] 
plt.plot(plot_depths, train_scores, marker='o', label='Training Accuracy', color='blue')
plt.plot(plot_depths, val_scores, marker='o', label='Validation Accuracy', color='red')

plt.title('Task 3: Accuracy vs Tree Depth (Bias-Variance Tradeoff)')
plt.xlabel('Max Depth (Complexity)')
plt.ylabel('Accuracy Score')
plt.xticks(plot_depths, [str(d) for d in depths])
plt.legend()
plt.grid(True)
plt.savefig('../plots/dt_bias_variance.png')
print("\nPlot saved to plots/dt_bias_variance.png")