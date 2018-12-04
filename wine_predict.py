"""
Complete tutorial: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
"""

import sklearn
print(sklearn.__version__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
dataset_url= 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.head(5))
print(data.shape)
print(data.describe())
y = data.quality
X= data.drop('quality', axis=1)
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

"""
X_train_scaled = preprocessing.scale(X_train)
X_train_scaled
X_train_scaled.mean(axis=0)
X_train_scaled.std(axis=0)
"""
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled.mean(axis = 0)
X_train_scaled.std(axis=0)

X_test_scaled = scaler.transform(X_test)
X_test_scaled.mean(axis=0)
X_test_scaled.std(axis=0)

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

pipeline.get_params()
hyperparameters = {'randomforestregressor__max_features':['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth':[None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

print(clf.best_params_)
print(clf.refit)
y_pred = clf.predict(X_test)
r2_score(y_test,y_pred)
mean_squared_error(y_test,y_pred)

joblib.dump(clf, 'rf_regressor.pkl')
clf2=joblib.load('rf_regressor.pkl')
clf2.predict(X_test)

""" method #2 """

#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

wine = pd.read_csv('http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep= ";")

"""
wine.head()
wine.info()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
plt.show()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
plt.show()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)
plt.show()

"""

#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])

wine['quality'].value_counts()

sns.countplot(wine['quality'])
plt.show()

#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))

param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, y_train)

#Best parameters for our svc model
grid_svc.best_params_

#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))

#Cross validation score for random forest and SGD

rfc_eval = cross_val_score(estimator = rfc, X=X_train, y=y_train, cv=10)
rfc_eval.mean()














