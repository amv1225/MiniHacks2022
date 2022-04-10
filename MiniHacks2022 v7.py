# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:31:44 2022

@author: tmkso
"""
#panda method

import csv
from csv import writer, reader
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB

#gets files and splits train into a train and test
file = open("train.csv")
data_train = pd.read_csv("train.csv")

#encodes all features
csvreader = csv.reader(file)
header = next(csvreader)

le = preprocessing.LabelEncoder()
for col in data_train:
    data_train[col]=le.fit_transform(data_train[col])
    data_train[col].unique()

#selects which features to remove
remove = ['grade', 'ID', 'address', 'paid', 'nursery', 'romantic','famrel','freetime','absences']
remove2 = ['ID', 'address', 'paid', 'nursery', 'romantic','famrel','freetime','absences']

#splits data
x = data_train.drop(remove, axis=1)
y = data_train['grade']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.05, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#different model creations
rfc = RandomForestClassifier(random_state=0,n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("Accuracy: rfc", mean_squared_error(pred_rfc, y_test))


clf = LogisticRegression(random_state=0,max_iter=5000)
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print("Accuracy: clf", mean_squared_error(pred_clf, y_test))

clr = LinearRegression()
clr.fit(X_train, y_train)
pred_clr = clr.predict(X_test)
print("Accuracy: clr", mean_squared_error(pred_clr, y_test))

ga = GaussianNB()
ga.fit(X_train, y_train)
pred_ga = ga.predict(X_test)
print("Accuracy: ga", mean_squared_error(pred_ga, y_test))

ad = AdaBoostClassifier(n_estimators=100)
ad.fit(X_train, y_train)
pred_ad = ad.predict(X_test)
print("Accuracy: ad", mean_squared_error(pred_ad, y_test))

########Submission Creation########

data_test = pd.read_csv("test.csv")

ID = data_test['ID'].copy(deep=True)

le = preprocessing.LabelEncoder()
for col in data_test:
    data_test[col]=le.fit_transform(data_test[col])
    data_test[col].unique()

fin = LinearRegression()
fin.fit(X_train, y_train)

x = data_test.drop(remove2, axis=1)
x = sc.fit_transform(x)

pred_fin = fin.predict(x)
final = np.around(pred_fin)

list = {'ID': ID, 'grade': final}  
       
df = pd.DataFrame(list)

print("Prediction:\n", df)

df.to_csv('SubmissionLRS.csv', index=False)  
