#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:05:51 2020

@authors: Soumya Dutta and Subhasish Bose
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from string import punctuation 
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

 
def read_jsonl_to_dataFrame(filepath,dfColname1,dfColname2,dfColname3):
    """
    THis is the method to read the train and test jsonl files into pandas dataframe
    parameter1: filepath of jsonl file
    parameter2: dataframe column name 1
    parameter3: dataframe column name 1
    parameter4: dataframe column name 1
    return the entire pandas dataframe
    """
    new_list = []
    with open(filepath, 'r') as json_file:
        #with open('./data/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        new_list.append(json.loads(json_str))
    df = pd.DataFrame(new_list,columns=(dfColname1,dfColname2,dfColname3))
    return df



def final_prediction_calculation(pred_response_context):
    """
    Method to predict the resu;t ie. SARCASM or NOT_SARCASM on given test data
    parameter 1: Single Cleaned Combined response and Context as input  
    Return Type: Returned the predicted output ie. SARCASM or NOT_SARCASM
    """
    svc_classification_result = ''.join(text_clf_lr.predict([pred_response_context]))
    return svc_classification_result


def write_prediction_results_in_list(df,response_context):
     """
     The method iteratively call final_prediction_calculation function to write the predicted
     output into the pandas data frame
     Parameter 1: Pandas dataframe 
     Parameter 2: Pandas dataframe header info. eg. "response + context" column 
     Return Type: List of all the predicted output
     """
     prediction_list_final = list()
     df_list = df[response_context].values.tolist()
     for review in df_list:
        final_prediction = 0       
        final_prediction = final_prediction_calculation(review)
        prediction_list_final.append(final_prediction)
     return prediction_list_final


        
def simple_feature_engieering_and_data_cleansing(df,df_header,stop_words):
    """
    Method to do some data cleaning steps after obsering the traing and test data set
    Parameter 1: Pandas Data Frame
    Parameter 2: Data Frame Columns
    Parameter 3: Stopwords selection 
    Return Type: Pandas DataFrame
    
    """
    
    df[df_header] = df[df_header].str.lower()
    df[df_header] = df[df_header].replace({'@USER': 'AT_USER'}, regex=True)
    df[df_header] = df[df_header].replace({'<URL>': 'URL'}, regex=True)
    df[df_header] = df[df_header].replace({'((www\.[^\s]+)|(https?://[^\s]+))': 'URL'}, regex=True)
    df[df_header] = df[df_header].replace({r'#([^\s]+)': r'\1'}, regex=True)
    df[df_header] = df[df_header].replace({r'@[^\s]+': 'AT_USER'}, regex=True)
    df[df_header] = df[df_header].replace({r'@[^\s]+': 'AT_USER'}, regex=True)
    df[df_header] = df[df_header].replace({r']': ''}, regex=True)   
    df[df_header] = df[df_header].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df[df_header] = df[df_header].str.replace('[^\x00-\x7F]','')
    df[df_header] = df[df_header].str.replace('\.', '')
    df[df_header] = df[df_header].str.replace('\->', '')
    df[df_header] = df[df_header].str.replace('[', '')
    
    df[df_header] = df[df_header].str.replace("'","") 
    df[df_header] = df[df_header].str.replace('"',"") 
    df[df_header] = df[df_header].str.replace('?',"") 
    df[df_header] = df[df_header].str.replace('!',"") 
    df[df_header] = df[df_header].str.replace(',',"") 
    df[df_header] = df[df_header].apply(lambda x: ' '.join([word.strip() for word in x.split()]))
    return df


# Read and store the training and test data into pandas dataframe

df_train = read_jsonl_to_dataFrame('../data/train.jsonl',"label","response","context")
df_test = read_jsonl_to_dataFrame('../data/test.jsonl',"id","response","context")
print("Training data DataFrame -->",df_train.head())
print("Test data DataFrame -->",df_test.head())


# Combine Response and Context Tweets in the dataframe both in training and test data

df_train["response + context"] = df_train["response"] + " " +  df_train["context"].astype(str) + " "
df_test["response + context"] = df_test["response"] + " " +  df_test["context"].astype(str) + " "


stopwords_train = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
stopwords_test = list(punctuation) + ['AT_USER','URL']


#Perform simple data cleaning steps after observing the data

df_train = simple_feature_engieering_and_data_cleansing(df_train,"response + context",stopwords_train)
df_test = simple_feature_engieering_and_data_cleansing(df_test,"response + context",stopwords_test)


X = df_train["response + context"] 
y = df_train["label"] 


#Traing and Test Step Split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=0)
 

#Create the model Pipe Line
# During test ing the performance of the model differnt hyperparameters of the Logistic REgression algorithm
# TF-IDF Vectorizer parameters are tuned so that model can provide the best perfomance
# Test size is kept very small because we have a seperate test data set available and we need 
#to predict the output based on the seperate test data 

text_clf_lr = Pipeline([('tfidf', TfidfVectorizer(max_features =20000,min_df=1,max_df=0.5, binary=1, use_idf=1, smooth_idf=1, sublinear_tf=1,ngram_range=((1,3)))),
                      ('clf', LogisticRegression(class_weight='balanced',solver='newton-cg',C=1)),
])

 
text_clf_lr.fit(X_train, y_train)  
 
# Form a prediction set
predictions = text_clf_lr.predict(X_test)

print("Classification Result for Logistic Regression\n", metrics.classification_report(y_test,predictions))
print("Overall accuracy for Logistic Regression\n",metrics.accuracy_score(y_test,predictions))


# Call the write_prediction_results_in_list method to create the prediction list with output SARCASM or NOT_SARCASM

prediction_list_final = write_prediction_results_in_list(df_test,"response + context")

df_test["Predicted_label"] = prediction_list_final
df_test_label_predictions = df_test[['id','Predicted_label']]

#Write the output to answer.txt file for auto grader
df_test_label_predictions.to_csv('../answer.txt',header=False, index=False, sep=',')
    








