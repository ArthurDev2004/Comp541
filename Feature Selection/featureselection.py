import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# load data from csv file 
insuranceData = pd.read_csv('insurance.csv')


# convert all categorical data to numerical data, because random forsts work better with them 

newInsuranceData = pd.get_dummies(insuranceData, columns=['sex', 'smoker', 'region'], drop_first=True)

# get rid of the target value
features = newInsuranceData.drop('charges', axis=1)
target = newInsuranceData['charges']

# get some data to be training data and other to be testing data 
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)

# get started on the random forest 
randomForest = RandomForestRegressor(n_estimators=100, random_state=42)

randomForest.fit(features_train, target_train)


# now we get the feature importances, which will help with the selection
featureImportances = randomForest.feature_importances_

featureImportanceData = pd.DataFrame({'feature' : features_train.columns, 'importance' : featureImportances}).sort_values(by='importance', ascending=False)

print("Features based on importance from the model: ")
print(featureImportanceData)