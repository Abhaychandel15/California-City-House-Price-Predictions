import os
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
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attr,cat_attr):
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

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # i have to train my model
    # 1. Load the dataset
    housing= pd.read_csv("housing.csv")

    # 2. Create a Stratified Test Set
    housing["income_cat"]= pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(housing,housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv",index=False)
    
        housing = housing.loc[train_index].drop("income_cat", axis=1)
    
    # 4. Separate Features and Labels
    housing_labels= housing["median_house_value"].copy()
    housing_features= housing.drop("median_house_value",axis=1)

    # 5.separate numerical and categorial columns
    num_attr= housing_features.drop("ocean_proximity",axis=1).columns.to_list()
    cat_attr= ["ocean_proximity"]
        
    pipeline=build_pipeline(num_attr,cat_attr)
    # 7. Transform the data
    housing_prep= pipeline.fit_transform(housing_features)
    print(housing_prep)
    
    model= RandomForestRegressor()
    model.fit(housing_prep,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is Trained Congrats!")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data= pd.read_csv("input.csv")
    transform_data= pipeline.transform(input_data)
    predictions= model.predict(transform_data)
    input_data["median_house_value"]=predictions

    input_data.to_csv("output.csv",index=False)
    print("inference is complete output saved to output.csv enjoy!!")