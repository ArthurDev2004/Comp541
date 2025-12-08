"""
Visualization script for data cleaning results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('insurance_cleaned.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Insurance Data Cleaning and Transformation Results', fontsize=16, fontweight='bold')

# 1. Age Groups Distribution
ax1 = axes[0, 0]
age_counts = df['age_group'].value_counts().sort_index()
ax1.bar(age_counts.index, age_counts.values, color='skyblue', edgecolor='black')
ax1.set_title('Age Groups Distribution', fontweight='bold')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. BMI Categories Distribution
ax2 = axes[0, 1]
bmi_counts = df['bmi_category'].value_counts()
ax2.bar(range(len(bmi_counts)), bmi_counts.values, color='lightcoral', edgecolor='black')
ax2.set_title('BMI Categories Distribution', fontweight='bold')
ax2.set_xlabel('BMI Category')
ax2.set_ylabel('Count')
ax2.set_xticks(range(len(bmi_counts)))
ax2.set_xticklabels(bmi_counts.index, rotation=45, ha='right')

# 3. Charges Ranges Distribution
ax3 = axes[0, 2]
charges_counts = df['charges_range'].value_counts()
ax3.bar(range(len(charges_counts)), charges_counts.values, color='lightgreen', edgecolor='black')
ax3.set_title('Charges Ranges Distribution', fontweight='bold')
ax3.set_xlabel('Charges Range')
ax3.set_ylabel('Count')
ax3.set_xticks(range(len(charges_counts)))
ax3.set_xticklabels(charges_counts.index, rotation=45, ha='right')

# 4. Original vs Capped Charges
ax4 = axes[1, 0]
ax4.scatter(df['charges'], df['charges_capped'], alpha=0.5, color='purple')
ax4.plot([df['charges'].min(), df['charges'].max()], 
         [df['charges'].min(), df['charges'].max()], 
         'r--', lw=2, label='No change line')
ax4.set_title('Original vs Capped Charges', fontweight='bold')
ax4.set_xlabel('Original Charges')
ax4.set_ylabel('Capped Charges')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Normalized Data Distribution (Standard)
ax5 = axes[1, 1]
normalized_cols = ['age_standardized', 'bmi_standardized', 'charges_standardized']
for col in normalized_cols:
    ax5.hist(df[col], alpha=0.5, label=col.replace('_standardized', ''), bins=30)
ax5.set_title('Standard Normalized Distributions', fontweight='bold')
ax5.set_xlabel('Standardized Value')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Normalized Data Distribution (Min-Max)
ax6 = axes[1, 2]
minmax_cols = ['age_minmax', 'bmi_minmax', 'charges_minmax']
for col in minmax_cols:
    ax6.hist(df[col], alpha=0.5, label=col.replace('_minmax', ''), bins=30)
ax6.set_title('Min-Max Normalized Distributions', fontweight='bold')
ax6.set_xlabel('Min-Max Value (0-1)')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_cleaning_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to: data_cleaning_visualization.png")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\n1. DISCRETIZATION SUMMARY:")
print(f"   Age Groups: {df['age_group'].value_counts().to_dict()}")
print(f"   BMI Categories: {df['bmi_category'].value_counts().to_dict()}")
print(f"   Charges Ranges: {df['charges_range'].value_counts().to_dict()}")
print(f"   Children Groups: {df['children_group'].value_counts().to_dict()}")

print("\n2. NORMALIZATION STATISTICS:")
print("\n   Standard Normalization (Z-score):")
for col in normalized_cols:
    print(f"     {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, "
          f"min={df[col].min():.6f}, max={df[col].max():.6f}")

print("\n   Min-Max Normalization (0-1):")
for col in minmax_cols:
    print(f"     {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, "
          f"mean={df[col].mean():.6f}")

print("\n3. OUTLIER HANDLING:")
capped_count = (df['charges'] != df['charges_capped']).sum()
print(f"   Charges capped: {capped_count} values")
print(f"   Original charges range: [{df['charges'].min():.2f}, {df['charges'].max():.2f}]")
print(f"   Capped charges range: [{df['charges_capped'].min():.2f}, {df['charges_capped'].max():.2f}]")
print(f"   BMI range after capping: [{df['bmi'].min():.2f}, {df['bmi'].max():.2f}]")

print("\n" + "=" * 80)

