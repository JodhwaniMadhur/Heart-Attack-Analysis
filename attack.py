import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,Lasso,ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

data=pd.read_csv("heart.csv")
print(data.info())
print(data.describe())

data['oldpeak']=data['oldpeak'].astype(int)
print(data.info())
print(data.head())

y=data['output'].values
data.drop('output',inplace=True,axis=1)
data.drop('fbs',inplace=True,axis=1)
data.drop('restecg',inplace=True,axis=1)
data.drop('sex',inplace=True,axis=1)
X=data

######################################################

rf=RandomForestClassifier(random_state=42)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=42,train_size=0.8)
rf.fit(X_train,Y_train)
prediction=rf.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print(ans*100)

######################################################

dtc=DecisionTreeClassifier(random_state=42)
dtc.fit(X_train,Y_train)
prediction=dtc.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print(ans*100)

######################################################

bg = BaggingClassifier(RandomForestClassifier(), max_samples= 100, max_features = 1.0, n_estimators = 40,n_jobs=-1,bootstrap=True)
bg.fit(X_train,Y_train)
print("Testing accuracy using bagging classifier : ",bg.score(X_test,Y_test)*100)
print("Training accuracy using bagging classifier : ",bg.score(X_train,Y_train)*100)


######################################################

vc=VotingClassifier(voting='hard',estimators=[('c1',rf),('c2',dtc),('c3',bg)])
vc.fit(X_train,Y_train)
prediction=vc.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print("Voting Classifier Accuracy is: ",ans*100)

######################################################

lr=LogisticRegression(max_iter=500,random_state=42)
lr.fit(X_train,Y_train)
lr.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print("Logistic Regression accuracy is: ",ans*100)

######################################################

sv=SVC(max_iter=300)
sv.fit(X_train,Y_train)
sv.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print("SVC Regression accuracy is: ",ans*100)

######################################################

adb=AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=0.5,n_estimators=100)
adb.fit(X_train,Y_train)
adb.predict(X_test)
ans=accuracy_score(Y_test,prediction)
print("Adaboost accuracy is: ",ans*100)

######################################################





def plot_feature_importance(model):
    plt.figure(figsize=(13,8))
    n_features=10
    plt.barh(range(n_features),model.feature_importances_,align='center')
    heart_features=[x for i,x in enumerate(data.columns)if i!=10]
    plt.yticks(np.arange(n_features),heart_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
    plt.show()

#plot_feature_importance(rf)