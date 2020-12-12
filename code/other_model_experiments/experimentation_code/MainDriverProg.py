#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 01:32:58 2020

@author: soumyadutta
"""

import preprocess
import NonNeturalNetwork
import NeturalNetwork
import pickle
from datetime import datetime
import pandas as pd

from keras.models import load_model
import NeturalNetwork
#import preprocess


import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


NeuralModelCrateFresh = False
NonNeuralNetwork = True
NonNeuralNetworkModelSave = True 

df_train = preprocess.read_jsonl_to_dataFrame('/users/soumyadutta/desktop/CourseProject/data/train.jsonl',"label","response","context")
print("Training data DataFrame -->",df_train.head())
df_test = preprocess.read_jsonl_to_dataFrame('/users/soumyadutta/desktop/CourseProject/data/test.jsonl',"id","response","context")
print("Test data DataFrame -->",df_test.head())

#df_train["Combined Context"] = df_train["response"] + df_train["context"].astype(str)
df_train["Combined Context"] = df_train["response"] 
df_test["Combined Context"] = df_test["response"] 


def final_prediction_calculation(model_naive_baise,model_svc,model_logistic_regression,model_votting_essence,model_BERT,pred_response_context):
    
    naive_baise_classification_result = ''.join(model_naive_baise.predict([pred_response_context]))
    svc_classification_result = ''.join(model_svc.predict([pred_response_context]))
    logistic_classification_result = ''.join(model_svc.predict([pred_response_context]))
    votting_classification_result = ''.join(model_votting_essence.predict([pred_response_context]))
    bert_classification_result = NeturalNetwork.predict_sarcasm(model_BERT,pred_response_context)
    
    prediction_list = []
    prediction_list.append(naive_baise_classification_result)
    prediction_list.append(svc_classification_result)
    prediction_list.append(logistic_classification_result)
    prediction_list.append(votting_classification_result)
    prediction_list.append(bert_classification_result)
    
    #print("FInal_Model_Prediction_list",','.join(prediction_list))
    
    if prediction_list.count("SARCASM") >= 3:
        return "SARCASM"
    if prediction_list.count("NOT_SARCASM") >= 3:
        return "NOT_SARCASM"

    return None


def write_prediction_results_in_list(df,response_context):
     prediction_list_final = list()
     #print("Hello Soumya Dutta")
     df_list = df[response_context].values.tolist()
     #print("df_list---->",df_list[0:100])
     #print("df List Length ------->",len(df_list))
     for review in df_list:
        final_prediction = 0
        #print("Review is-->",review)
        final_prediction = final_prediction_calculation(model_nb, model_svc, model_lr, model_voteclf, bert_model_load,review)
        #print("Final Values of predictions,,,,,,",final_prediction)
        prediction_list_final.append(final_prediction)
        
    
     return prediction_list_final


if NonNeuralNetwork:
    clean_response = preprocess.CleanTokenize(df_train,"Combined Context")
    #clean_response = ' '.join([str(elem) for elem in clean_response]) 
    print(clean_response[0:10])
    clean_response_test = preprocess.CleanTokenize(df_test,"Combined Context")
    df_train["cleaned_response"] = clean_response
    df_test["cleaned_response"] = clean_response_test 
    
    myreview = "@USER @USER @USER Sending Blessings and big hugs to all you wonderful friends 💜 #Blessed #TmKindness <URL>@USER @USER @USER Sending Blessings 🙏 my lovely #friends #TmKindness ❤ Happy Tuesday dear Nena 💐 and Everyone 🌺 ❤ <URL>@USER @USER @USER Sending Blessings and big hugs to all you wonderful friends 💜 #Blessed #TmKindness <URL>"
    if NonNeuralNetworkModelSave:
        NonNeturalNetwork.nonDeepLearningModel(df_train)
    #load the file
    model_nb = pickle.load(open('model_nb.sav', 'rb'))
    model_svc = pickle.load(open('model_svc.sav', 'rb'))
    model_lr = pickle.load(open('model_lr.sav', 'rb'))
    #model_random = pickle.load(open('model_randomforest.sav', 'rb'))
    model_voteclf = pickle.load(open('model_vottingclf.sav', 'rb'))
    bert_model_load = load_model('myfirstmodel.h5')
    is_sarcasm = NeturalNetwork.predict_sarcasm(bert_model_load,myreview)
    
    print("Naive Baise my review ",''.join(model_nb.predict([myreview])))  # be sure to put "myreview" inside square brackets
    print("svc my review ",''.join(model_svc.predict([myreview])))  # be sure to put "myreview" inside squar    
    print("logistic Regression my review ",''.join(model_lr.predict([myreview])))  # be sure to put "myreview" inside squar  
    #print("Random Forest Regression my review ",''.join(model_random.predict([myreview])))  # be sure to put "myreview" inside squar  
    print("Votting Essnece Classifier my review ",''.join(model_voteclf.predict([myreview])))
    print("Neural BERT model Prediction",is_sarcasm)
    
    # final_prediction = final_prediction_calculation(model_nb, model_svc, model_lr, model_voteclf, bert_model_load,myreview)
    # print("FInal MOdel Prediction Output--->",final_prediction)
    
    
    
    # start=datetime.now()
    # print("start time is",start)
    # prediction_list_final = write_prediction_results_in_list(df_train,"cleaned_response")
    # #print("predicted list final------>",prediction_list_final)
    # df_train["Predicted_label"] = prediction_list_final
    # print("end time is",datetime.now())
    # print("Total Time Taken to predict the training set--->",datetime.now() - start)
    
    
    # df_train.to_csv("df_train_predictions.csv",index=False)
    
    
    start=datetime.now()
    print("start time is",start)
    prediction_list_final = write_prediction_results_in_list(df_test,"cleaned_response")
    #print("predicted list final------>",prediction_list_final)
    df_test["Predicted_label"] = prediction_list_final
    print("end time is",datetime.now())
    print("Total Time Taken to predict the training set--->",datetime.now() - start)
    df_test.to_csv("df_test_predictions.csv",index=False)
    df_test_label_predictions = df_test[['id','Predicted_label']]
    df_test_label_predictions.to_csv('answer.txt',header=False, index=False, sep=',')
    
    
    
    
    
    
if NeuralModelCrateFresh:
    #df_train["Combined Context"] = df_train["response"] + df_train["context"].astype(str) 
    clean_response = preprocess.CleanTokenizeforNeuralNetwork(df_train,"response")
    df_train["cleaned_response"] = clean_response
    df_train["label"] = df_train.apply(lambda x: 1  if x["label"] =="SARCASM" else 0,axis=1)
    print(clean_response[0:10])


    validation_split = 0.1
    max_list, max_length = preprocess.find_max_length(clean_response)

    print("Maximum List",max_list)
    print("Maximum Lenght of list",max_length)
    NeturalNetwork.bidirectionalModelCrationAndSave(df_train,max_length,validation_split,clean_response)