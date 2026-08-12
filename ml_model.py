# ML code for the model
# We are gonna use two or three models like
# logistic regression
#random forest
#decision tree
# SVM
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("dataset_cleaned.csv")
df.head() 
X=df.drop(columns=["addicted_label"])
y=df["addicted_label"]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)
scaler=StandardScaler()
train_x_scaled=scaler.fit_transform(train_x)
randomForest=RandomForestClassifier(n_jobs=-1,max_depth=12)
model=randomForest.fit(train_x, train_y)
pred=model.predict(test_x[:5]).round(0)
print(pred)
print(test_y[:5])
print(model.score(X,y))
logestic=LogisticRegression(max_iter=100)
model2=logestic.fit(train_x,train_y)
pred2=model.predict(test_x[:5])
print(test_y)
print(model2.score(X,y))
print(model.get_params())
print(model2.get_params())
svm=LinearSVC()
model3=logestic.fit(train_x,train_y)
pred3=model.predict(test_x[:5])
print(test_y)
print(model3.score(X,y))
