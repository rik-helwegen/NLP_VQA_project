"""
CBOW

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

from collections import defaultdict
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import h5py
import math
import json
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import re
import dill as pickle
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

#FUNCTIONS
# one HOT function
def one_hot_encoding(data):
    embeddings = np.zeros((len(data), nwords))
    for i, question in enumerate(data):
        for word in question:
            embeddings[i][word] = 1
    return embeddings

# dataloader Functions
def dataLoader():
    # QUESTIONS AND ANSWERS LOADING #
    with open("data_processed/x_test.pkl", "rb") as fp:   #Pickling
        x_test = pickle.load(fp)
    with open("data_processed/y_test.pkl", "rb") as fp:   #Pickling
        y_test = pickle.load(fp)
    with open("data_processed/y_test_type.pkl", "rb") as fp:   #Pickling
        y_test_type = pickle.load(fp)

    #size of data used
    # test_size = len(x_test)
    # test_size = 2962

    # use size for shorter time
    # x_test = x_test[:test_size]
    img_ids_test = [x[0] for x in x_test]
    questions_test = [x[1] for x in x_test]

    # test
    # answers_test = y_test[:test_size]
    answers = [x for x in y_test]
    answer_type = [x for x in y_test_type]

    # encodes to one hot vector
    questions_test = one_hot_encoding(questions_test)

    # combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
    test_data = [[questions_test[i], answers[i], img_ids_test[i], answer_type[i]] for i in range(len(questions_test))]
    test_data = np.asarray(test_data)

    return test_data

#CLASSES
# neural network
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, image_features_dim, output_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.embedding_output = nn.Linear(embedding_dim, output_dim)
        self.img_output = nn.Linear(image_features_dim, output_dim)

    def forward(self, question_input, image_input):
        embeds = self.embedding(question_input)
        embedding_output = self.embedding_output(embeds)
        img_output = self.img_output(image_input)
        addition = torch.add(embedding_output, img_output)

        return addition, embedding_output, img_output

## DATA PART ##
# image features loading
path_to_h5_file = 'data/VQA_image_features.h5'
path_to_json_file = 'data/VQA_img_features2id.json'
# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])
# load mapping file
with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

# load vocabularies
with open("data_processed/w2i.pkl", "rb") as fp:   #Pickling
    w2i = pickle.load(fp)
with open("data_processed/t2i.pkl", "rb") as fp:   #Pickling
    t2i = pickle.load(fp)
with open("data_processed/x_test_question.pkl", "rb") as fp:   #Pickling
    questions = pickle.load(fp)

#length vocabulary of questions, answers and image features
nwords = len(w2i)
ntags = len(t2i)
nfeatures = len(img_features[0])

# inverse dictionaries to get the answers back
i2w = {y:x for x,y in w2i.items()}
i2t = {y:x for x,y in t2i.items()}

# load question and answer datasets
test_data = dataLoader()

##NETWORK PART
# after which number of epochs we want a evaluation:
validation_update = 1

# set parameters for hyper optimization
minibatch_size = 128
vocwords = nwords
voctags = ntags
vocfeatures = nfeatures
print(nwords, ntags, nfeatures)

#Lowest loss model is used for analysis
#load model with lowest test loss
model.load_state_dict(torch.load('model/cbow_lowest_loss.pt'))
which_model = "loss"

#load model with highest accuracy
# model.load_state_dict(torch.load('model/cbow_highest_acc.pt'))
# which_model = "accuracy"

"""Evaluate a model on a data set."""
test_loss = 0.0

# index 0 -> total, index 1 -> yes/no, index2 -> other, index3 -> number
correct = [0.0, 0.0, 0.0, 0.0]
correct_five = [0.0, 0.0, 0.0, 0.0]

# total amount of answer type
total_other = 0
total_yesno = 0
total_number = 0

#array for position of answer on prediction sorted on score
position_array = np.zeros(voctags)
position_arrayYN = np.zeros(voctags)
position_arrayOT = np.zeros(voctags)
position_arrayNU = np.zeros(voctags)

#loop through all indexes test data set
for i in range(len(test_data)):
    # get question, answer, img_id and answer_type on index
    question = test_data[i][0]
    answer = test_data[i][1]
    img_id = test_data[i][2]
    answer_type = test_data[i][3]

    #get image features from image id
    image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

    #create 2 tensors for image features and question
    question_tensor = Variable(torch.FloatTensor([question]))
    image_features_tensor = Variable(torch.FloatTensor([image]))

    #get total score, question score and image score from model
    score, score_q, score_i = model(question_tensor, image_features_tensor)

    #set loss -> get correct answer -> calculate loss -> add loss to test loss
    loss = nn.CrossEntropyLoss()
    target = Variable(torch.LongTensor([answer]))
    output = loss(score, target)
    test_loss += output.data[0]

    # check prediction from output network
    predict = score.data.numpy().argmax(axis=1)[0]

    # get an list with tuples of index of word and score
    predict_sorted = []
    index = 0
    for pre in score.data[0]:
        predict_sorted.append([index,pre])
        index +=1
    predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
    predict_sorted = np.asarray(predict_sorted)[:,0].tolist()

    # if(img_id == 27055 or img_id == 552089 or img_id == 15514 or img_id == 395155):
    if(predict_sorted.index(answer) < 1):
        print("     QUESTION     ", questions[i][0], "  answer = ", i2t[answer])
        # loop three times to get the three highest predictions
        for i in range(3):
            # check prediction from output network
            predict = score.data.numpy().argmax(axis=1)[0]
            print("Prediction = ",i2t[predict], "  img id = ", img_id)
            print("total    score =   %.6f" % score.data[0][predict])
            print("question score =   %.6f" % score_q.data[0][predict])
            print("image    score =   %.6f" % score_i.data[0][predict])
            score.data[0][predict] = -1E10

        #Create the highest ranked outcomes with only question input
        predict_sorted = []
        index = 0
        for pre in score_q.data[0]:
            predict_sorted.append([index,pre])
            index +=1
        predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
        predict_sorted = np.asarray(predict_sorted)[:,0].tolist()
        words =  [i2t[word] for word in predict_sorted[0:6]]
        score =  [score_q.data[0][int(word)] for word in predict_sorted[0:6]]

        #Get the top 6 from this ranked list
        print("top 6 questions")
        for i in range(6):
            print (i+1, "=", words[i], " -> ", score[i])

        #create the highest ranked outcomes with only image input
        predict_sorted = []
        index = 0
        for pre in score_i.data[0]:
            predict_sorted.append([index,pre])
            index +=1
        predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
        predict_sorted = np.asarray(predict_sorted)[:,0].tolist()
        words =  [i2t[word] for word in predict_sorted[0:6]]
        score =  [score_i.data[0][int(word)] for word in predict_sorted[0:6]]

        # get the top 6 from this list
        print("top 3 image")
        for i in range(6):
            print (i+1, "=", words[i], " -> ", score[i])
