"""
Insurance Data Cleaning and Transformation
Performs:
1. Fill null values
2. Find and handle outliers
3. Discretize data
4. Normalize data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("=" * 80)
print("INSURANCE DATA CLEANING AND TRANSFORMATION")
print("=" * 80)

df = pd.read_csv('insurance.csv')
print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nOriginal data types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")

# ============================================================================
# 1. CHECK AND FILL NULL VALUES
# ============================================================================
print("\n" + "=" * 80)
print("1. NULL VALUE ANALYSIS")
print("=" * 80)

null_counts = df.isnull().sum()
print(f"\nNull values per column:\n{null_counts}")

if null_counts.sum() > 0:
    print("\nFilling null values...")
    # Fill numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - {col}: filled with median = {median_val:.2f}")
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"  - {col}: filled with mode = {mode_val}")
else:
    print("\nNo null values found in the dataset!")

# ============================================================================
# 2. FIND OUTLIERS
# ============================================================================
print("\n" + "=" * 80)
print("2. OUTLIER DETECTION")
print("=" * 80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    outliers = data[z_scores > threshold]
    return outliers

numerical_cols = ['age', 'bmi', 'children', 'charges']
outlier_summary = {}

print("\nOutlier Analysis (IQR Method):")
print("-" * 80)
for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    print(f"\n{col.upper()}:")
    print(f"  Lower bound: {lower:.2f}")
    print(f"  Upper bound: {upper:.2f}")
    print(f"  Number of outliers: {outlier_count} ({outlier_percentage:.2f}%)")
    
    if outlier_count > 0:
        print(f"  Min outlier value: {outliers[col].min():.2f}")
        print(f"  Max outlier value: {outliers[col].max():.2f}")
    
    outlier_summary[col] = {
        'count': outlier_count,
        'percentage': outlier_percentage,
        'lower_bound': lower,
        'upper_bound': upper,
        'outliers': outliers
    }

# Decision on outliers
print("\n" + "-" * 80)
print("OUTLIER HANDLING DECISION:")
print("-" * 80)

# For charges: Cap outliers (insurance charges can be very high but should be capped)
if outlier_summary['charges']['count'] > 0:
    print("\nCHARGES: Capping outliers at 99th percentile")
    charges_99th = df['charges'].quantile(0.99)
    df['charges_capped'] = df['charges'].clip(upper=charges_99th)
    print(f"  Original max: {df['charges'].max():.2f}")
    print(f"  Capped max: {df['charges_capped'].max():.2f}")
    print(f"  Values capped: {(df['charges'] > charges_99th).sum()}")

# For BMI: Cap extreme outliers (BMI > 50 or < 10 are likely errors)
if outlier_summary['bmi']['count'] > 0:
    print("\nBMI: Capping extreme values (< 10 or > 50)")
    bmi_outliers = df[(df['bmi'] < 10) | (df['bmi'] > 50)]
    if len(bmi_outliers) > 0:
        print(f"  Extreme BMI values found: {len(bmi_outliers)}")
        df['bmi'] = df['bmi'].clip(lower=10, upper=50)
        print(f"  BMI range after capping: [{df['bmi'].min():.2f}, {df['bmi'].max():.2f}]")

# For age: Keep outliers (age can legitimately vary)
print("\nAGE: Keeping outliers (legitimate variation)")

# For children: Keep outliers (some people may have many children)
print("\nCHILDREN: Keeping outliers (legitimate variation)")

# ============================================================================
# 3. DISCRETIZE DATA
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA DISCRETIZATION")
print("=" * 80)

# Discretize age into age groups
print("\nDiscretizing AGE into age groups:")
age_bins = [0, 25, 35, 45, 55, 65, 100]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
print(f"  Age groups created: {df['age_group'].value_counts().to_dict()}")

# Discretize BMI into categories
print("\nDiscretizing BMI into categories:")
bmi_bins = [0, 18.5, 25, 30, 35, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, include_lowest=True)
print(f"  BMI categories created: {df['bmi_category'].value_counts().to_dict()}")

# Discretize charges into cost ranges
print("\nDiscretizing CHARGES into cost ranges:")
charges_bins = [0, 5000, 10000, 20000, 30000, 100000]
charges_labels = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
df['charges_range'] = pd.cut(df['charges'], bins=charges_bins, labels=charges_labels, include_lowest=True)
print(f"  Charges ranges created: {df['charges_range'].value_counts().to_dict()}")

# Discretize children into groups
print("\nDiscretizing CHILDREN into groups:")
df['children_group'] = pd.cut(df['children'], 
                               bins=[-0.5, 0.5, 2.5, 5.5, 10], 
                               labels=['No Children', '1-2 Children', '3-5 Children', '5+ Children'],
                               include_lowest=True)
print(f"  Children groups created: {df['children_group'].value_counts().to_dict()}")

# ============================================================================
# 4. NORMALIZE DATA
# ============================================================================
print("\n" + "=" * 80)
print("4. DATA NORMALIZATION")
print("=" * 80)

# Create a copy for normalization
df_normalized = df.copy()

# Normalize numerical columns using StandardScaler (Z-score normalization)
print("\nStandard Normalization (Z-score) for numerical columns:")
scaler_standard = StandardScaler()
numerical_cols_to_normalize = ['age', 'bmi', 'children', 'charges']

for col in numerical_cols_to_normalize:
    if col in df_normalized.columns:
        df_normalized[f'{col}_standardized'] = scaler_standard.fit_transform(df_normalized[[col]])
        print(f"  - {col}_standardized: mean={df_normalized[f'{col}_standardized'].mean():.6f}, "
              f"std={df_normalized[f'{col}_standardized'].std():.6f}")

# Normalize using Min-Max scaling (0-1 range)
print("\nMin-Max Normalization (0-1 range) for numerical columns:")
scaler_minmax = MinMaxScaler()

for col in numerical_cols_to_normalize:
    if col in df_normalized.columns:
        df_normalized[f'{col}_minmax'] = scaler_minmax.fit_transform(df_normalized[[col]])
        print(f"  - {col}_minmax: min={df_normalized[f'{col}_minmax'].min():.6f}, "
              f"max={df_normalized[f'{col}_minmax'].max():.6f}")

# ============================================================================
# SUMMARY AND SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nFinal dataset shape: {df_normalized.shape}")
print(f"\nColumns in final dataset:")
for i, col in enumerate(df_normalized.columns, 1):
    print(f"  {i:2d}. {col}")

# Save cleaned and transformed data
output_file = 'insurance_cleaned.csv'
df_normalized.to_csv(output_file, index=False)
print(f"\n✓ Cleaned and transformed data saved to: {output_file}")

# Save a summary report
summary_file = 'data_cleaning_summary.txt'
with open(summary_file, 'w') as f:
    f.write("INSURANCE DATA CLEANING AND TRANSFORMATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. NULL VALUES:\n")
    f.write(f"   - Original null count: {null_counts.sum()}\n")
    f.write(f"   - After filling: {df.isnull().sum().sum()}\n\n")
    
    f.write("2. OUTLIERS:\n")
    for col, info in outlier_summary.items():
        f.write(f"   - {col}: {info['count']} outliers ({info['percentage']:.2f}%)\n")
    f.write("\n")
    
    f.write("3. DISCRETIZATION:\n")
    f.write(f"   - Age groups: {len(age_labels)} categories\n")
    f.write(f"   - BMI categories: {len(bmi_labels)} categories\n")
    f.write(f"   - Charges ranges: {len(charges_labels)} categories\n")
    f.write(f"   - Children groups: 4 categories\n\n")
    
    f.write("4. NORMALIZATION:\n")
    f.write("   - Standard normalization (Z-score) applied\n")
    f.write("   - Min-Max normalization (0-1) applied\n")
    f.write(f"   - Normalized columns: {len(numerical_cols_to_normalize) * 2}\n")

print(f"✓ Summary report saved to: {summary_file}")

print("\n" + "=" * 80)
print("DATA CLEANING AND TRANSFORMATION COMPLETE!")
print("=" * 80)

