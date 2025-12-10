import kagglehub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# dataset

path = kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset")
print("Path to dataset files:", path)
csv_path = path + "/insurance.csv"
dataset = pd.read_csv(csv_path)

X = dataset.drop("charges", axis=1)
y = dataset["charges"]

# train/test split

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# categorical/  numerical split

categoricalfeatures = ["sex", "smoker", "region"]
numericalfeatures = ["age", "bmi", "children"]

# model

preprocessing = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(), categoricalfeatures),
        ("numerical", StandardScaler(), numericalfeatures),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("model", GradientBoostingRegressor(random_state=23)),
    ]
)

# fit

cv_scores = cross_val_score(pipeline, Xtrain, ytrain, cv=5, scoring="r2")
pipeline.fit(Xtrain, ytrain)
ypredtrain = pipeline.predict(Xtrain)
ypredtest = pipeline.predict(Xtest)

# metrics

trainrmse = mean_squared_error(ytrain, ypredtrain) ** 0.5
testrmse = mean_squared_error(ytest, ypredtest) ** 0.5
trainmae = mean_absolute_error(ytrain, ypredtrain)
testmae = mean_absolute_error(ytest, ypredtest)
trainr2 = r2_score(ytrain, ypredtrain)
testr2 = r2_score(ytest, ypredtest)

# print

print("model performance:")
print("train RMSE:", trainrmse)
print("test RMSE:", testrmse)
print("train MAE:", trainmae)
print("test MAE:", testmae)
print("train R²:", trainr2)
print("test R²:", testr2)
print("CV R² scores:", cv_scores)
print("mean CV R²:", cv_scores.mean())

