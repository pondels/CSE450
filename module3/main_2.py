#%%
import pandas as pd
from datetime import datetime
import seaborn as sns
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import datetime as date
from sklearn.metrics import mean_absolute_error
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
#%%
main_df =         pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout_df =      pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test.csv')
mini_holdout_df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
#%%
main_df["Year Sold"] =         [int(main_df['date'].loc[i][:4])         for i in range(len(main_df['date']))]
holdout_df["Year Sold"] =      [int(holdout_df['date'].loc[i][:4])      for i in range(len(holdout_df['date']))]
mini_holdout_df["Year Sold"] = [int(mini_holdout_df['date'].loc[i][:4]) for i in range(len(mini_holdout_df['date']))]

main_df["Month Sold"] =         [int(main_df['date'].loc[i][4:6])         for i in range(len(main_df['date']))]
holdout_df["Month Sold"] =      [int(holdout_df['date'].loc[i][4:6])      for i in range(len(holdout_df['date']))]
mini_holdout_df["Month Sold"] = [int(mini_holdout_df['date'].loc[i][4:6]) for i in range(len(mini_holdout_df['date']))]

main_df["Day Sold"] =         [int(main_df['date'].loc[i][6:8])         for i in range(len(main_df['date']))]
holdout_df["Day Sold"] =      [int(holdout_df['date'].loc[i][6:8])      for i in range(len(holdout_df['date']))]
mini_holdout_df["Day Sold"] = [int(mini_holdout_df['date'].loc[i][6:8]) for i in range(len(mini_holdout_df['date']))]
holdout_df

##%

main_df["price_per_sqft"] = main_df["price"] / main_df[""]


#%%
main_df = pd.get_dummies(main_df, drop_first=True)
main_df



#%%





#Model After this section 
#%%
XGBoost = xgb.XGBRegressor(max_depth = 4, random_state=11, n_estimators=500)

# X = main_df[['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
#        'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
#        'sqft_living15', 'sqft_lot15', 'Date New']]

# X = main_df[['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
#        'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
#        'sqft_living15', 'Year Sold', 'Month Sold', 'Day Sold']]
X = main_df.drop(['price'], axis=1)

y = main_df['price']

# ros = RandomOverSampler(random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# xee, yee = ros.fit_resample(X_train, y_train)

XGBoost.fit(X_train, y_train)

#%%
predictions = XGBoost.predict(X_test)
mean_squared_error(y_test, predictions, squared=False)
#%%
x_holdout = holdout_df[['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'Year Sold', 'Month Sold', 'Day Sold']]

predictions = XGBoost.predict(x_holdout)
y_mini = pd.DataFrame()
y_mini['price'] = predictions
y_mini.to_csv('holdout-module3-team4-predictions.csv', index=False)
#%%

