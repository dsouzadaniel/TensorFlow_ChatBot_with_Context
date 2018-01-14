#################################################
#################################################
## Author : Daniel D'souza
## Email : ddsouza@umich.edu
#################################################
#################################################

###### Libraries #######
import numpy as np
import nltk
import random
import tensorflow as tf
import tflearn
import json
import pickle


## Importing the Lancaster Stemmer fron NLTK to stem our words
from nltk.stem.lancaster import  LancasterStemmer
stemmer = LancasterStemmer()

## Import Our Intent File( Our Chatbot Brain)
with open('intents.json') as json_data:
    intents = json.load(json_data)

#################################################
#### PART 1 : Unpacking Intent and Data Prep ####
#################################################

## Code to unpack the Intents file and use its contents
# All possible words in your questions
words = []
# A tuple of 'words associated' and 'categories' for all questions
documents = []
# All possible categories that your chatbot can carry a conversation in
classes = []
# Ignoring these words
ignore_words = ['?']

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in your questions list(patterns)
        w = nltk.word_tokenize(pattern)
        # Add it to all possible words list
        words.extend(w)
        # Add the Tuple of (w,tag) to documents list
        documents.append((w,intent['tag']))
        # Also add it to your classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


## Code to prepare data
# Lowercasing, Stemming and Deduping your word list
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Deduping your classes
classes = sorted(list(set(classes)))

# Print an Update of the Information collected
print("*"*50)
print(len(documents), " Documents ")
print(len(classes), " Classes ", classes)
print(len(words)," Words ", words)
print("*"*50)


## Data Munging
# Creating Training and Output Lists
training = []
output = []

# Creating an empty array for our output
output_empty = [0]*len(classes)

# Converting our queriers into BOW format
for doc in documents:
    # doc is now a tuple of tokenized 'words associated' and 'category'
    # Create an empy bag
    bag = []
    # Get your tokenized words out
    pattern_words = doc[0]
    # Stem them
    pattern_words = [stemmer.stem(word) for word in pattern_words ]
    # Create a BOW representation
    for w in words:
        # 1 for every present word, 0 for every absent one
        bag.append(1) if w in pattern_words else bag.append(0)

    # Create your respective output vector
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append this newly created training input and output sample into your training list
    training.append([bag, output_row])

# Shuffle your Trainining Features, coz why not?
random.shuffle(training)

# Package it into an Numpy Representation for easy compatibility
training = np.array(training)

# Split your training list into input x and output y
train_x = list(training[:,0])
train_y = list(training[:,1])

#################################################
#### PART 2 : Training the Chatbot Model ####
#################################################

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

## Start Training ( Use the Gradient Descent Algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

print(" Model is Trained!")
## Save our model and pickle the data_structures for the Chatbot to use
model.save('model.tflearn')

pickle.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open("train_data", "wb"))

print(" Model Saved + Pickles Pickled!")

