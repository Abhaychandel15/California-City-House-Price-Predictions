import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
housing= pd.read_csv("housing.csv")

# 2. Create a Stratified Test Set
housing["income_cat"]= pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set  = housing.loc[test_index].drop("income_cat", axis=1)


# 3. we will work on the copy of the data
housing= strat_train_set.copy()

# 4. Separate Features and Labels
housing_labels= housing["median_house_value"].copy()
housing= housing.drop("median_house_value",axis=1)

# 5.separate numerical and categorial columns
num_attr= housing.drop("ocean_proximity",axis=1).columns.to_list()
cat_attr= ["ocean_proximity"]

# 6. Lets make the pipe line
# for numerical columns
num_pipeline= Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
# for categorial columns
cat_pipeline= Pipeline([
    ("encoder",OneHotEncoder(handle_unknown="ignore")),
])
# construct Full pipeline

full_pipeline= ColumnTransformer([
    ("num",num_pipeline,num_attr),
    ("cat",cat_pipeline,cat_attr)
])

# 7. Transform the data
housing_prep= full_pipeline.fit_transform(housing)

# print(housing_prep.shape)

# 8. Train the Model

# Linear regression Model
linear_reg= LinearRegression()
linear_reg.fit(housing_prep,housing_labels)
linear_preds= linear_reg.predict(housing_prep)
# linear_rmse= root_mean_squared_error(housing_labels,linear_preds)
# print(f"The root mean squared error for Linear Regression is {linear_rmse}")
linear_rmses= -cross_val_score(linear_reg,housing_prep,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(linear_rmses).describe())

# Decision Tree Regressor Model
dec_reg= DecisionTreeRegressor()
dec_reg.fit(housing_prep,housing_labels)
dec_preds= dec_reg.predict(housing_prep)
# dec_rmse= root_mean_squared_error(housing_labels,dec_preds)
desc_rmses= -cross_val_score(dec_reg,housing_prep,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root mean squared error for DecisionTree Regressor is {desc_rmse}")
print(pd.Series(desc_rmses).describe())

# Random Forest Regressor Model
random_reg= RandomForestRegressor()
random_reg.fit(housing_prep,housing_labels)
random_preds= random_reg.predict(housing_prep)
# random_rmse= root_mean_squared_error(housing_labels,random_preds)
# print(f"The root mean squared error for RandomForest Regressor is {random_rmse}")
random_rmses= -cross_val_score(random_reg,housing_prep,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(random_rmses).describe())

