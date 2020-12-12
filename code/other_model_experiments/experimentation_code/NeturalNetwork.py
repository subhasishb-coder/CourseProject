#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:48:51 2020

@author: soumyadutta
"""

import preprocess
import pandas as pd
import numpy as np
import os
from sklearn import metrics

import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def predict_sarcasm(new_model,s,max_length=548):
    '''
    This is the main main method to predict the tweet is sarcastic or not based on Bidirectional LSTM
    If score >50 it will treat as sarcasm..otherwise not sarcasm
    '''
    x_final = pd.DataFrame({"response":[s]})
    test_lines = preprocess.CleanTokenizeforNeuralNetwork(x_final,"response")
    #print("Lines will be",test_lines)
    
    tokenizer_obj = Tokenizer()
    #tokenizer_obj.fit_on_texts(clean_response)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    
    
    
    #test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = new_model.predict(test_review_pad)
    pred*=100
    #print("pred is -->",pred)
    if pred[0][0]>=50: 
        return "SARCASM" 
    else: 
        return "NOT SARCASM"




def bidirectionalModelCrationAndSave(df_train,max_length,validation_split,clean_response):
    """
    Main method for implemeting Bidirectional LSTM
    For this binary classification problem to determine sarcasm or not sarcasm we have used Glove twitter (glove.twitter.27B.100d.txt)word embedding 
    We have tried with different Epocs but found that data converges at 2 epocs
    """
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(clean_response)
    sequences = tokenizer_obj.texts_to_sequences(clean_response)

    word_index = tokenizer_obj.word_index
    print("unique tokens - ",len(word_index))
    vocab_size = len(tokenizer_obj.word_index) + 1
    print('vocab size -', vocab_size)

    lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
    sentiment =  df_train['label'].values

    indices = np.arange(lines_pad.shape[0])
    np.random.shuffle(indices)
    lines_pad = lines_pad[indices]
    sentiment = sentiment[indices]

    num_validation_samples = int(validation_split * lines_pad.shape[0])

    X_train_pad = lines_pad[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    X_test_pad = lines_pad[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]



    print('Shape of X_train_pad:', X_train_pad.shape)
    print('Shape of y_train:', y_train.shape)

    print('Shape of X_test_pad:', X_test_pad.shape)
    print('Shape of y_test:', y_test.shape)


    embeddings_index = {}
    embedding_dim = 100
    GLOVE_DIR = "/users/soumyadutta/desktop/CourseProject/Glove_twitter"
    f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))


    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    c = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            c+=1
            embedding_matrix[i] = embedding_vector
    print(c)


    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=True)



    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

    print('Summary of the built model...')
    print(model.summary())

    history = model.fit(X_train_pad, y_train, batch_size=32, epochs=2, validation_data=(X_test_pad, y_test), verbose=2)
    model.save('myfirstmodel.h5')



    # Plot results
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()



    print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test_pad,y_test)[1]*100)
    pred = model.predict_classes(X_test_pad)
    print(metrics.classification_report(y_test, pred, target_names = ['Not Sarcastic','Sarcastic']))


    print("Confusion Matrix\n", metrics.confusion_matrix(y_test,pred))









