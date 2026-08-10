# ML code for the model
# We are gonna use two or three models like
# logistic regression
#random forest
#decision tree
# SVM
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("dataset_cleaned.csv")
df.head() 
X=df.drop(columns=["addicted_label"])
y=df["addicted_label"]
train_x,train_y,test_x,test_y=train_test_split(X,y,test_size=0.2,random_state=0)
scaler=StandardScaler()
train_x_scaled=scaler.fit_transform(train_x)
randomForest=RandomForestRegressor()
model=randomForest.fit(train_x_scaled, test_x)