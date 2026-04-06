# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor



"""
Predicting Distributions - pd3
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 probabilistic predictions
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from distribution_regressor import DistributionRegressor

# Učitavanje stvarnih loto podataka iz CSV (bez random/sintetičkih podataka)
np.random.seed(39)
csv_path = "/Users/4c/Desktop/GHQ/data/loto7hh_4592_k27.csv"
df = pd.read_csv(csv_path)
cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
draws = df[cols].values.astype(float)

# Za deo vizualizacije koristimo Num1 kao cilj (skalar), kao i originalni primer
X = draws[:-1]           # prethodna kombinacija
Y = draws[1:]            # sledeća kombinacija
y = Y[:, 0]              # Num1
feature_cols = [f"f{i+1}" for i in range(7)]
X_df = pd.DataFrame(X, columns=feature_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=39
)

print("Training model...")
model = DistributionRegressor(
    n_estimators=200,
    learning_rate=0.1,
    verbose=0,
    random_state=39
)

model.fit(X_train, y_train)

# Select a few test points to visualize
n_examples = 6
example_idx = np.random.choice(len(X_test), n_examples, replace=False)
X_examples = X_test.iloc[example_idx]
y_examples = y_test[example_idx]

# Get predicted distributions
y_grid = np.linspace(y_test.min() - 20, y_test.max() + 20, 200)
grids_ex, distributions, offsets_ex = model.predict_distribution(X_examples)

# Get point predictions and intervals
y_pred = model.predict(X_examples)
interval_90 = model.predict_interval(X_examples, confidence=0.90)
lower, upper = interval_90[:, 0], interval_90[:, 1]

# Visualize distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(n_examples):
    ax = axes[i]
    
    # Plot the distribution
    ax.fill_between(grids_ex[i], distributions[i], alpha=0.3, color='blue', label='Distribution')
    ax.plot(grids_ex[i], distributions[i], 'b-', linewidth=2)
    
    # Mark key points
    ax.axvline(y_examples[i], color='green', linestyle='--', linewidth=2, label=f'True: {y_examples[i]:.1f}')
    ax.axvline(y_pred[i], color='red', linestyle='-', linewidth=2, label=f'Mean: {y_pred[i]:.1f}')
    ax.axvline(lower[i], color='orange', linestyle=':', linewidth=1.5, label='90% interval')
    ax.axvline(upper[i], color='orange', linestyle=':', linewidth=1.5)
    
    # Styling
    ax.set_xlabel('y value')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Example {i+1}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Highlight the area between interval bounds
    mask = (grids_ex[i] >= lower[i]) & (grids_ex[i] <= upper[i])
    ax.fill_between(grids_ex[i][mask], distributions[i][mask], alpha=0.2, color='orange')

plt.suptitle('Predicted Probability Distributions for Individual Test Points', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/4c/Desktop/GHQ/kurzor/DistributionRegressor-main/examples/distribution_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'distribution_visualization.png'")

# Sample from distributions
print("\n" + "="*60)
print("Sampling from Predicted Distributions")
print("="*60)

# sample_y nije dostupan u DistributionRegressorCDF; uzorkovanje radimo ručno iz PMF-a
samples = []
grids_s, dists_s, _ = model.predict_distribution(X_examples[:3])
rng = np.random.default_rng(39)
for i in range(3):
    probs = dists_s[i] / np.sum(dists_s[i])
    samples.append(rng.choice(grids_s[i], size=1000, p=probs))
samples = np.array(samples)

for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"  True value:      {y_examples[i]:7.2f}")
    print(f"  Mean prediction: {y_pred[i]:7.2f}")
    print(f"  Sample mean:     {samples[i].mean():7.2f}")
    print(f"  Sample std:      {samples[i].std():7.2f}")
    print(f"  Sample range:    [{samples[i].min():7.2f}, {samples[i].max():7.2f}]")

print("\n" + "="*60)
print("Key Insight")
print("="*60)
print("Each prediction is a full probability distribution, not just a point!")
print("This enables:")
print("  • Uncertainty quantification")
print("  • Risk assessment")
print("  • Multi-modal predictions")
print("  • Confidence-based decision making")

print("\n" + "="*60)
print("Loto 7/39 - Predikcija sledeće kombinacije")
print("="*60)

# Predikcija sledeće kompletne kombinacije 7/39 (po pozicijama)
X_full = pd.DataFrame(draws[:-1], columns=feature_cols)
Y_full = draws[1:]
X_next = pd.DataFrame(draws[-1:].astype(float), columns=feature_cols)

predicted_next = []
for i in range(7):
    m = DistributionRegressor(
        n_estimators=200,
        learning_rate=0.1,
        verbose=0,
        random_state=39
    )
    m.fit(X_full, Y_full[:, i])
    p = float(m.predict(X_next)[0])
    predicted_next.append(int(np.clip(np.rint(p), 1, 39)))

predicted_next = np.array(predicted_next, dtype=int)
print("Predicted next loto 7/39 combination:", predicted_next)

"""
Training model...
✓ Saved visualization to 'distribution_visualization.png'

============================================================
Sampling from Predicted Distributions
============================================================

Example 1:
  True value:         3.00
  Mean prediction:    5.67
  Sample mean:        5.65
  Sample std:         4.13
  Sample range:    [   1.00,   22.49]

Example 2:
  True value:         1.00
  Mean prediction:    3.99
  Sample mean:        3.97
  Sample std:         3.09
  Sample range:    [   1.00,   23.04]

Example 3:
  True value:         7.00
  Mean prediction:    5.08
  Sample mean:        5.18
  Sample std:         4.02
  Sample range:    [   1.00,   24.14]

============================================================
Key Insight
============================================================
Each prediction is a full probability distribution, not just a point!
This enables:
  • Uncertainty quantification
  • Risk assessment
  • Multi-modal predictions
  • Confidence-based decision making

============================================================
Loto 7/39 - Predikcija sledeće kombinacije
============================================================
Predicted next loto 7/39 combination: 
[ 5  9 15 20 24 30 35]
"""
