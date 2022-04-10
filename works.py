# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:31:44 2022
@author: tmkso
"""
#panda method
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

file = open("train.csv")
data_train = pd.read_csv("train.csv")
data_train, data_test = train_test_split(data_train, test_size=0.2, random_state=0)
csvreader = csv.reader(file)
header = next(csvreader)
'''
print(type(df))
print(df.grade.describe())
'''

key = pd.read_csv("test.csv")

def encode_features(df_train, df_test):
    features = header
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
data_train, data_test = encode_features(data_train, data_test)
print(data_train.head())

X = data_train.drop(['ID', 'absences'], axis=1)
y = data_train['grade']

clf = GaussianNB()

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf.fit(X, y)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')


print("Prediction: ")
print(clf.predict(X))
print("Accuracy", clf.score(X, y))

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))