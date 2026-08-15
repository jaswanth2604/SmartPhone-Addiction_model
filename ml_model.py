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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
df=pd.read_csv("dataset_cleaned.csv")
df.head() 
#train test split
X=df.drop(columns=["addicted_label"])
y=df["addicted_label"]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)
scaler=StandardScaler()
train_x_scaled=scaler.fit_transform(train_x)
test_x_scaled=scaler.fit_transform(test_x)
#random forest
randomForest=RandomForestClassifier(n_jobs=-1,max_depth=12)
model=randomForest.fit(train_x, train_y)
pred=model.predict(test_x[:5]).round(0)
pred_prob=model.predict_proba(test_x[:5])
print(pred)
print(test_y[:5])
print(model.score(X,y))
#logestic regression
logestic=LogisticRegression(max_iter=100)
model2=logestic.fit(train_x,train_y)
pred2=model.predict(test_x[:5])
print(test_y)
print(model2.score(X,y))
print()
print(model.get_params())
print(model2.get_params())
#linearSVC()
svm=LinearSVC()
model3=logestic.fit(train_x,train_y)
pred3=model.predict(test_x[:5])
print(test_y)
print(model3.score(X,y))
mean_absolute_error(train_y[:5],pred3)
mean_squared_error(train_y[:5], pred)
param_grid=[{"n_estimators":[10,20,30,40,50,60,70,80,90,100],"max_depth":[6,8,10,12,14,16]}]
gridSearch=GridSearchCV(model, param_grid,cv=5)
cv=gridSearch.fit(X,y)
df2=pd.DataFrame(cv.cv_results_)
print(cv.best_score_)
print(cv.best_params_)
cross_score= cross_val_score(model, X[:1000],y[:1000],scoring="neg_mean_squared_error",cv=3)
print(cross_score)
#using the polinomial features in random forest
#rforest_pipeline=Pipeline([("polynomials",PolynomialFeatures(degree=2,include_bias=False)),
#                           ("model",RandomForestClassifier(n_jobs=-1,max_depth=12))])
#rforest_pipeline.fit(train_x, train_y)
#pred_pline=rforest_pipeline.predict(test_x[:5]).round(0)
#print(pred_pline)
#print(test_y[:5])
#print(model.score(X,y))