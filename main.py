#################################################
#################################################
## Author : Daniel D'souza
## Email : ddsouza@umich.edu
#################################################
#################################################

###### Libraries #######
import tflearn
import tensorflow as tf
import numpy as np
import pickle
import json
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

## Restore the Trained model and the Data Structures

data = pickle.load( open("train_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

## Import our ChatBot Intents File
with open('intents.json') as json_data:
    intents = json.load(json_data)

## Load the Model
# Reset the Underlying Graph Data
tf.reset_default_graph()

## Network Architechture
# Input Layer : BOW_sizex1
net = tflearn.input_data(shape=[None, len(train_x[0])])
# Fully_Connected Layer(FC1) : 8x1
net = tflearn.fully_connected(net, 8)
# Fully_Connected Layer(FC2) : 8x1
net = tflearn.fully_connected(net, 8)
# Fully_Connected Output(FC1) : Classes_sizex1 : (Softmax Activated)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# Regression
net = tflearn.regression(net)

## Define Model and Tensorboard Setup for Visualization
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')


#################################################
#### The Response and Classificaton Code  ####
#################################################
#
# We need 4 functions here to complete our ChatBot Functionality:
# 1. clean_up_sent() : To clean up your sentences by lowercasing and stemming
# 2. create_bow() : To return a BOW representation of the query to input into the trained model for classification
# 3. classify() : To perform the actual classification on the sentence
# 4. response(): Tie everything up tidily and print out a response
#
#################################################

# Error Threshold
ERROR_THRESHOLD = 0.25

## Context Dictionary
context = {}


def clean_up_sent(sentence):
    # Tokenize your sentence
    sentence_words = nltk.word_tokenize(sentence, language='english')
    # Stem each word
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words

def create_bow(sentence, words,show_details=False):
    # Tokenize and Stem the Query
    sentence_words = clean_up_sent(sentence)

    # BOW Representation
    bag = [0]*len(words)

    for query_word in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: {0}".format(w))

    return np.array(bag)

def classify(sentence):
    # Generate the Prediction of the Query
    results = model.predict([bow(sentence, words)])[0]
    # Filter out predictions beyond your previously defined Error Threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # Return tuple of Intent and Probability
    return return_list

def response(sentence, userID='123', show_details=True):
    results = classify(sentence)

    #Find your matching intent tag
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    # now return a random answer for that category
                    if 'context_set' in i:
                        context[userID] = 'context_set'
                        if show_details:
                            print('context:', i['context_set'])

                    if('context_filter' not in i or (userID in context and 'context_filter' in i \
                                                             and context[userID] == i['context_filter'])):
                        return print(random.choice(i['responses']))

            results.pop(0)




