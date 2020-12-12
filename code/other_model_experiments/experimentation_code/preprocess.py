import pandas as pd
import json
from sklearn.model_selection import train_test_split
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics


# import re
# import matplotlib.pyplot as plt
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
# from keras.layers.embeddings import Embedding
# from keras.initializers import Constant
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


"""
This test has been performed w/o doing any modifications in the training and test data
Overall accuracy is about 70%.. There are many False Positives as well
We need to think more how to increase the model accurracy after revisting the data once more
"""


"""
Read the Training and Test Data of jsonl format to pandas dataframe
Read training data
"""

def read_jsonl_to_dataFrame(filepath,dfColname1,dfColname2,dfColname3):
    new_list = []
    with open(filepath, 'r') as json_file:
        #with open('./data/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        new_list.append(json.loads(json_str))
    df = pd.DataFrame(new_list,columns=(dfColname1,dfColname2,dfColname3))
    return df
    



def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text


def CleanTokenizeforNeuralNetwork(df,response):
    head_lines = list()
    lines = df[response].values.tolist()

    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        #head_lines.append(words)
        head_lines.append(words)
    return head_lines


def CleanTokenize(df,response):
    head_lines = list()
    lines = df[response].values.tolist()
    #lines_new = df["context"].values.tolist()
    #lines.extend(lines_new)
    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        #head_lines.append(words)
        head_lines.append(' '.join(str(ele) for ele in words))
    return head_lines



def find_max_length(lst):
    maxList = max(lst, key = len) 
    maxLength = max(map(len, lst))
    return maxList, maxLength
    
    



#print(df_train.isnull().sum())  
#print(df_test.isnull().sum())          

