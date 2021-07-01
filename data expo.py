import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\apollo\Downloads\cars data\car data.csv")
df_final = df.drop('Car_Name',axis="columns")
df_final["no_year"] = 2021 - df_final["Year"]
df_final.drop("Year",axis=1,inplace=True)
df_final = pd.get_dummies(df_final, drop_first=True)
Y = df_final.iloc[:,0]
X = df_final.iloc[:,1:]      # or X = df_final.drop("Selling_Price", axis=1)

import sklearn
from sklearn.model_selection import train_test_split as tts

X_train,X_test,Y_train,Y_test = tts(X,Y,test_size=0.25,random_state=5)
df_final.to_csv('df_final_car.csv')
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_train,Y_train)
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_, index = X.columns)
feat_imp.nlargest(5).plot(kind='barh')

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)
from sklearn.metrics import mean_absolute_error as mae , mean_squared_error as mse , r2_score as r2
print("MAE:",mae(Y_test,Y_pred))
print("MSE:",mse(Y_test,Y_pred))
print("RMSE:",np.sqrt(mse(Y_test,Y_pred)))
print("R2:",r2(Y_test,Y_pred))
import pickle
file = open('car_price_pridiction_model.pkl','wb')

pickle.dump(rf, file)