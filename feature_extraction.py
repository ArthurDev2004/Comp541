import pandas as pd
import numpy as np

# Loads the cleaned data (output from data_cleaning_analysis.py)
df = pd.read_csv('insurance_cleaned.csv')

print("=" * 80)
print("FEATURE EXTRACTION STARTED")
print("=" * 80)
print(f"Loaded cleaned data with shape: {df.shape}")
print("Columns:", list(df.columns))
print()

# One-hot encode categorical features
# use the cleaned categorical columns already created:
#   - sex, smoker, region (original categorical features)
#   - age_group, bmi_category, children_group (discretized groupings)
categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category', 'children_group']

print("One-hot encoding categorical columns:")
print(" ", categorical_cols)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"After one-hot encoding, shape: {df_encoded.shape}")
print()

# Interaction features
# Example: interaction between BMI and smoking status
# expect a column like 'smoker_yes' after get_dummies
if 'smoker_yes' in df_encoded.columns:
    df_encoded['bmi_x_smoker'] = df_encoded['bmi'] * df_encoded['smoker_yes']
    print("Created interaction feature: bmi_x_smoker (bmi * smoker_yes)")

# Another example: age * bmi (combined effect of age and body mass)
df_encoded['age_x_bmi'] = df_encoded['age'] * df_encoded['bmi']
print("Created interaction feature: age_x_bmi (age * bmi)")
print()

# Log-transform charges to reduce skewness
# use charges_capped (from cleaning) if present, otherwise use charges
target_col = 'charges_capped' if 'charges_capped' in df_encoded.columns else 'charges'

df_encoded['log_' + target_col] = np.log1p(df_encoded[target_col])
print(f"Created log-transformed target: log_{target_col}")
print()

# then save the feature-engineered dataset
output_file = 'insurance_features.csv'
df_encoded.to_csv(output_file, index=False)

print("=" * 80)
print("FEATURE EXTRACTION COMPLETE")
print(f"Feature-engineered dataset saved to: {output_file}")
print(f"Final shape: {df_encoded.shape}")
print("=" * 80)
