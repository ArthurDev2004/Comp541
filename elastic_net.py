import kagglehub
import pandas as pd
import numpy as np
import joblib  # <--- THIS IS REQUIRED
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
print("Downloading dataset...")
path = kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset")
csv_path = path + "/insurance.csv"
dataset = pd.read_csv(csv_path)

# 2. Feature Engineering
dataset['smoker_code'] = dataset['smoker'].map({'yes': 1, 'no': 0})
dataset['bmi_smoker_interaction'] = dataset['bmi'] * dataset['smoker_code']

X = dataset.drop(columns=['charges', 'smoker_code'])
y = dataset['charges']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children', 'bmi_smoker_interaction']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(random_state=42, max_iter=5000))
])

# 5. Training
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0],
    'regressor__l1_ratio': [0.5, 0.7, 0.9, 1.0]
}

print("Training Elastic Net Model...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 6. Evaluation
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# 7. DEPLOYMENT (This creates the missing file)
joblib.dump(best_model, 'elastic_net_model.pkl') 
print("SUCCESS: 'elastic_net_model.pkl' has been saved!")

# Save text results too
output_text = f"Elastic Net Results\nTest R2: {test_r2:.4f}"
with open('elastic_net_summary.txt', 'w') as f:
    f.write(output_text)