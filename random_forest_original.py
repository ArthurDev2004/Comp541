"""
Random Forest Regressor on Original Insurance Dataset
Using minimal feature selection and extraction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD ORIGINAL DATASET
# ============================================================================
print("=" * 80)
print("RANDOM FOREST ON ORIGINAL INSURANCE DATASET")
print("=" * 80)

df = pd.read_csv('insurance.csv')

print(f"\nOriginal Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nBasic Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# ============================================================================
# MINIMAL PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("MINIMAL PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop('charges', axis=1)
y = df['charges']

print(f"\nTarget Variable (charges):")
print(f"  Min: ${y.min():.2f}")
print(f"  Max: ${y.max():.2f}")
print(f"  Mean: ${y.mean():.2f}")
print(f"  Median: ${y.median():.2f}")
print(f"  Std Dev: ${y.std():.2f}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nOriginal Features:")
print(f"  Categorical: {categorical_cols}")
print(f"  Numerical: {numerical_cols}")

# Encode categorical variables (minimal encoding)
print(f"\nEncoding categorical variables...")
X_processed = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

print(f"\nProcessed Features:")
print(X_processed.head())

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================
print("\n" + "=" * 80)
print("RANDOM FOREST REGRESSOR")
print("=" * 80)

print("\nWhy Random Forest?")
print("- Ensemble of decision trees reduces variance")
print("- Handles non-linear relationships well")
print("- Robust to outliers")
print("- Works well with mixed data types")
print("- Provides feature importance naturally")
print("- Less prone to overfitting than single trees")

# Parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print(f"\nTotal combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
print("This will take approximately 5-10 minutes...")

# Grid search with cross-validation
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("\nTraining Random Forest with Grid Search...")
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

print(f"\n" + "=" * 60)
print("BEST PARAMETERS FOUND:")
print("=" * 60)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV Score (MSE): {-grid_search.best_score_:.2f}")
print(f"Best CV Score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Predictions
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nRandom Forest Performance:")
print(f"  Train RMSE: {train_rmse:.2f}")
print(f"  Test RMSE: {test_rmse:.2f}")
print(f"  Train MAE: {train_mae:.2f}")
print(f"  Test MAE: {test_mae:.2f}")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")

# Overfitting check
overfitting_score = train_r2 - test_r2
print(f"\nOverfitting Analysis:")
print(f"  Difference (Train R² - Test R²): {overfitting_score:.4f}")
if overfitting_score < 0.05:
    print(f"  ✓ Good generalization (low overfitting)")
elif overfitting_score < 0.1:
    print(f"  ⚠ Moderate overfitting")
else:
    print(f"  ✗ High overfitting detected")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Create main figure with 6 subplots
fig = plt.figure(figsize=(18, 12))

# 1. Feature Importance
ax1 = plt.subplot(2, 3, 1)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(feature_importance)))
bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontsize=11, fontweight='bold')
plt.title('Feature Importance\n(Random Forest)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(feature_importance['importance']):
    plt.text(v, i, f' {v:.4f}', va='center', fontsize=10)

# 2. Actual vs Predicted (Training)
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_train, y_train_pred, alpha=0.5, s=30, color='dodgerblue', edgecolors='darkblue', linewidth=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=3, label='Perfect Prediction', alpha=0.8)
plt.xlabel('Actual Charges ($)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Charges ($)', fontsize=11, fontweight='bold')
plt.title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}, RMSE = ${train_rmse:.2f}', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 3. Actual vs Predicted (Test)
ax3 = plt.subplot(2, 3, 3)
plt.scatter(y_test, y_test_pred, alpha=0.6, s=35, color='mediumseagreen', edgecolors='darkgreen', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=3, label='Perfect Prediction', alpha=0.8)
plt.xlabel('Actual Charges ($)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Charges ($)', fontsize=11, fontweight='bold')
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}, RMSE = ${test_rmse:.2f}', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 4. Residuals Plot (Training)
ax4 = plt.subplot(2, 3, 4)
train_residuals = y_train - y_train_pred
plt.scatter(y_train_pred, train_residuals, alpha=0.5, s=30, color='dodgerblue', edgecolors='darkblue', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=3, alpha=0.8)
plt.xlabel('Predicted Charges ($)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals ($)', fontsize=11, fontweight='bold')
plt.title(f'Training Set: Residuals\nMean Error = ${train_residuals.mean():.2f}', 
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

# 5. Residuals Plot (Test)
ax5 = plt.subplot(2, 3, 5)
test_residuals = y_test - y_test_pred
plt.scatter(y_test_pred, test_residuals, alpha=0.6, s=35, color='mediumseagreen', edgecolors='darkgreen', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=3, alpha=0.8)
plt.xlabel('Predicted Charges ($)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals ($)', fontsize=11, fontweight='bold')
plt.title(f'Test Set: Residuals\nMean Error = ${test_residuals.mean():.2f}', 
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6. Performance Metrics Comparison
ax6 = plt.subplot(2, 3, 6)
metrics = ['RMSE', 'MAE', 'R²']
train_values = [train_rmse, train_mae, train_r2]
test_values = [test_rmse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, train_values, width, label='Training', color='skyblue', edgecolor='darkblue', linewidth=1.5)
bars2 = plt.bar(x + width/2, test_values, width, label='Test', color='lightgreen', edgecolor='darkgreen', linewidth=1.5)

plt.xlabel('Metrics', fontsize=11, fontweight='bold')
plt.ylabel('Value', fontsize=11, fontweight='bold')
plt.title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
plt.xticks(x, metrics, fontsize=10)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('rf_original_dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: rf_original_dataset_analysis.png")
plt.show()

# Additional detailed visualizations
fig2 = plt.figure(figsize=(16, 6))

# 7. Error Distribution
ax7 = plt.subplot(1, 3, 1)
plt.hist(train_residuals, bins=50, alpha=0.7, label='Training', color='dodgerblue', edgecolor='darkblue', linewidth=1)
plt.hist(test_residuals, bins=50, alpha=0.7, label='Test', color='mediumseagreen', edgecolor='darkgreen', linewidth=1)
plt.xlabel('Prediction Error ($)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.axvline(x=0, color='red', linestyle='--', lw=2, alpha=0.8)
plt.grid(True, alpha=0.3, axis='y')

# 8. Tree Depth Impact (if available)
ax8 = plt.subplot(1, 3, 2)
# Get predictions from individual trees
tree_predictions_train = np.array([tree.predict(X_train) for tree in best_rf.estimators_])
tree_predictions_test = np.array([tree.predict(X_test) for tree in best_rf.estimators_])

# Calculate cumulative average predictions
cumulative_train_pred = np.cumsum(tree_predictions_train, axis=0) / np.arange(1, len(best_rf.estimators_) + 1)[:, np.newaxis]
cumulative_test_pred = np.cumsum(tree_predictions_test, axis=0) / np.arange(1, len(best_rf.estimators_) + 1)[:, np.newaxis]

# Calculate RMSE for each cumulative prediction
train_rmse_evolution = [np.sqrt(mean_squared_error(y_train, pred)) for pred in cumulative_train_pred]
test_rmse_evolution = [np.sqrt(mean_squared_error(y_test, pred)) for pred in cumulative_test_pred]

plt.plot(range(1, len(best_rf.estimators_) + 1), train_rmse_evolution, 
         label='Training RMSE', color='dodgerblue', linewidth=2.5, alpha=0.8)
plt.plot(range(1, len(best_rf.estimators_) + 1), test_rmse_evolution, 
         label='Test RMSE', color='mediumseagreen', linewidth=2.5, alpha=0.8)
plt.xlabel('Number of Trees', fontsize=12, fontweight='bold')
plt.ylabel('RMSE ($)', fontsize=12, fontweight='bold')
plt.title('RMSE vs Number of Trees\n(Convergence)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 9. Error by Actual Value
ax9 = plt.subplot(1, 3, 3)
plt.scatter(y_test, test_residuals, alpha=0.6, s=40, color='mediumseagreen', 
            edgecolors='darkgreen', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=2, alpha=0.8)
plt.xlabel('Actual Charges ($)', fontsize=12, fontweight='bold')
plt.ylabel('Prediction Error ($)', fontsize=12, fontweight='bold')
plt.title('Prediction Error vs Actual Value\n(Test Set)', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_original_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_original_detailed_analysis.png")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nPrediction Errors (Residuals):")
print(f"\n  Training Set:")
print(f"    Mean: ${train_residuals.mean():.2f}")
print(f"    Std Dev: ${train_residuals.std():.2f}")
print(f"    Min: ${train_residuals.min():.2f}")
print(f"    Max: ${train_residuals.max():.2f}")
print(f"    25th percentile: ${np.percentile(train_residuals, 25):.2f}")
print(f"    50th percentile (median): ${np.percentile(train_residuals, 50):.2f}")
print(f"    75th percentile: ${np.percentile(train_residuals, 75):.2f}")

print(f"\n  Test Set:")
print(f"    Mean: ${test_residuals.mean():.2f}")
print(f"    Std Dev: ${test_residuals.std():.2f}")
print(f"    Min: ${test_residuals.min():.2f}")
print(f"    Max: ${test_residuals.max():.2f}")
print(f"    25th percentile: ${np.percentile(test_residuals, 25):.2f}")
print(f"    50th percentile (median): ${np.percentile(test_residuals, 50):.2f}")
print(f"    75th percentile: ${np.percentile(test_residuals, 75):.2f}")

print(f"\nModel Configuration:")
print(f"  Number of Trees: {best_rf.n_estimators}")
print(f"  Max Depth: {best_rf.max_depth}")
print(f"  Min Samples Split: {best_rf.min_samples_split}")
print(f"  Min Samples Leaf: {best_rf.min_samples_leaf}")
print(f"  Max Features: {best_rf.max_features}")

print(f"\nPrediction Range:")
print(f"  Training Predictions: ${y_train_pred.min():.2f} to ${y_train_pred.max():.2f}")
print(f"  Test Predictions: ${y_test_pred.min():.2f} to ${y_test_pred.max():.2f}")
print(f"  Actual Training: ${y_train.min():.2f} to ${y_train.max():.2f}")
print(f"  Actual Test: ${y_test.min():.2f} to ${y_test.max():.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nFiles saved:")
print("  - rf_original_dataset_analysis.png")
print("  - rf_original_detailed_analysis.png")

