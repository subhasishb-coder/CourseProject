#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:09:37 2020

@author: soumyadutta
"""

import preprocess
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier 
import pickle


def nonDeepLearningModel(df_train):
    X = df_train['cleaned_response']
    y = df_train['label']
    #myreview = "@USER @USER @USER The craziest thing about his tweet is he is calling SOMEONE ELSE OUT for being racist ."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Naïve Bayes:
    text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                      ('clf', MultinomialNB()),
                      ])
    # Linear SVC:
    text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                      ('clf', LinearSVC()),
                      ])
    
    
    # Logistic Regression 
    text_clf_lr = Pipeline([('tfidf', TfidfVectorizer()),
                      ('clf', LogisticRegression()),
                      ])
    # # Random Forest Classifier
    # text_clf_ramdomforest = Pipeline([('tfidf', TfidfVectorizer()),
    #                   ('clf', RandomForestClassifier(n_estimators = 10)),
    #                   ])
    
    
    # Fit the Naive Baise Model    
    text_clf_nb.fit(X_train, y_train)   
    pickle.dump(text_clf_nb, open('model_nb.sav', 'wb'))
    # Form a prediction set
    predictions_nb = text_clf_nb.predict(X_test)   
    #Report the confusion matrix
    print("Confusion Matrix Result for Naive Baise\n", metrics.confusion_matrix(y_test,predictions_nb))
    # Print a classification report
    print("Classification Result for Naive Baise\n", metrics.classification_report(y_test,predictions_nb))   
    # Print the overall accuracy
    print("Overall accuracy for Naive Baise\n",metrics.accuracy_score(y_test,predictions_nb))
    
    # Fit the Linear Support Vector Classifier Model
    text_clf_lsvc.fit(X_train, y_train)
    pickle.dump(text_clf_lsvc, open('model_svc.sav', 'wb'))
    #Form a prediction set
    predictions_svc = text_clf_lsvc.predict(X_test)
    # Report the confusion matrix
    print("Confusion Matrix Result for Linear SVC\n", metrics.confusion_matrix(y_test,predictions_svc))
    # Print a classification report
    print("Classification Result for Linear SVC\n", metrics.classification_report(y_test,predictions_svc))   
    # Print the overall accuracy
    print("Overall accuracy for Linear SVC\n",metrics.accuracy_score(y_test,predictions_svc))
    
    text_clf_lr.fit(X_train, y_train)
    pickle.dump(text_clf_lr, open('model_lr.sav', 'wb'))
    predictions_lr = text_clf_lsvc.predict(X_test)
    # Report the confusion matrix
    print("Confusion Matrix Result for Linear Logistic Regression\n", metrics.confusion_matrix(y_test,predictions_lr))
    # Print a classification report
    print("Classification Result for Linear Logistic Regression\n", metrics.classification_report(y_test,predictions_lr))   
    # Print the overall accuracy
    print("Overall accuracy for Linear Logistic Regression\n",metrics.accuracy_score(y_test,predictions_lr))
    
    
    # text_clf_ramdomforest.fit(X_train, y_train)
    # pickle.dump(text_clf_ramdomforest, open('model_randomforest.sav', 'wb'))
    
    
    voting_clf = VotingClassifier(estimators=[('nb', text_clf_nb), ('svc', text_clf_lsvc), ('lr', text_clf_lr)],voting='hard')
    voting_clf.fit(X_train, y_train)
    pickle.dump(voting_clf, open('model_vottingclf.sav', 'wb'))
    # predictions_rmf = text_clf_ramdomforest.predict(X_test)
    # # Report the confusion matrix
    # print("Confusion Matrix Result for random Forest Regression\n", metrics.confusion_matrix(y_test,predictions_rmf))
    # # Print a classification report
    # print("Classification Result for random Forest Regression\n", metrics.classification_report(y_test,predictions_rmf))   
    # # Print the overall accuracy
    # print("Overall accuracy for Linear Random Forest Regression\n",metrics.accuracy_score(y_test,predictions_rmf))
    
    
    
    
    
    #return predictions_nb,predictions_svc,predictions_lr,text_clf_ramdomforest



    
    