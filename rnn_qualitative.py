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

    #split data for ease
    img_ids_test = [x[0] for x in x_test]
    questions_test = [x[1] for x in x_test]
    answers = [x for x in y_test]
    answer_type = [x for x in y_test_type]

    # combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
    test_data = [[questions_test[i], answers[i], img_ids_test[i], answer_type[i]] for i in range(len(questions_test))]
    test_data = np.asarray(test_data)

    return test_data

#CLASSES
# neural network
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, img_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.concat = nn.Linear(hidden_size + img_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.img_output = nn.Linear(img_size, num_classes)
    def forward(self, x, image, batch_size, hidden, q_length):
        embedded = self.embed(x)
        out, _ = self.lstm(embedded)
        out = torch.gather(out, 1, q_length.view(-1,1,1).expand(batch_size,1,hidden))
        concat = torch.cat((image, out[:, -1, :]), 1)
        image_with_hidden = self.concat(concat)
        image_with_hidden = self.relu(image_with_hidden)
        final = self.fc(image_with_hidden)

        #forward with images only
        imageOnly = Variable(torch.zeros(1,2176))
        imageOnly.data[0,:2048] = concat.data[0,:2048]
        imageOnly = self.concat(imageOnly)
        imageOnly = self.relu(imageOnly)
        imageOnly = self.fc(imageOnly)

        #forward with words only
        wordsOnly = Variable(torch.zeros(1,2176))
        wordsOnly.data[0,2048:] = concat.data[0,2048:]
        wordsOnly = self.concat(wordsOnly)
        wordsOnly = self.relu(wordsOnly)
        wordsOnly = self.fc(wordsOnly)

        return final, wordsOnly, imageOnly


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

#inverse dictionaries
i2w = {y:x for x,y in w2i.items()}
i2t = {y:x for x,y in t2i.items()}

# load question and answer datasets
test_data = dataLoader()

##NETWORK PART
# after which number of epochs we want a evaluation:
validation_update = 1

# set parameters for hyper optimization
minibatch_size = 128
vocwords = nwords+1
voctags = ntags
vocfeatures = nfeatures
print(nwords, ntags, nfeatures)

#create model
minibatch_size = 64
hidden_size = 128
embedding_size = 164
model = RNN(vocwords, embedding_size, nfeatures, hidden_size, 1, ntags)
model.load_state_dict(torch.load('model/rnnloss.pt'))

"""Evaluate a model on a data set."""
test_loss = 0.0

# index 0 -> total, index 1 -> yes/no, index2 -> other, index3 -> number
correct = [0.0, 0.0, 0.0, 0.0]
correct_five = [0.0, 0.0, 0.0, 0.0]

#questions of type
total_other = 0
total_yesno = 0
total_number = 0

#position of answer on prediction sorted on score
position_array = np.zeros(voctags)
position_arrayYN = np.zeros(voctags)
position_arrayOT = np.zeros(voctags)
position_arrayNU = np.zeros(voctags)

# forward with batch size = 1
for i in range(len(test_data)):
    # get question, answer, img_id and answer_type on index
    question = test_data[i][0]
    answer = test_data[i][1]
    img_id = test_data[i][2]
    answer_type = test_data[i][3]

    #Question input length for getting the right hidden state
    input_q_length = Variable(torch.LongTensor([len(question)-1]))

    #Image input
    image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

    #create 2 tensors for image features and question
    question_tensor = Variable(torch.LongTensor([question]))
    image_features_tensor = Variable(torch.FloatTensor([image]))

    #get total score, question score and image score from model
    score, score_q, score_i = model(question_tensor, image_features_tensor, 1, hidden_size, input_q_length)
    score =  score.view(1,-1)

    #calculate loss
    loss = nn.CrossEntropyLoss()
    target = Variable(torch.LongTensor([answer]))
    output = loss(score, target)
    test_loss += output.data[0]

    # check prediction from output network
    predict = score.data.numpy().argmax(axis=1)[0]
    predict_sorted = []
    index = 0
    for pre in score.data[0]:
        predict_sorted.append([index,pre])
        index +=1
    predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
    predict_sorted = np.asarray(predict_sorted)[:,0].tolist()

    #change value to < > for getting only top1, top5 or very low scoring answers
    if(predict_sorted.index(answer) > 0):
    # if(img_id == 446307):
        print("     QUESTION     ", questions[i][0], "  answer = ", i2t[answer])
        # loop three times to get the three highest predictions
        for i in range(3):
            # check prediction from output network
            predict = score.data.numpy().argmax(axis=1)[0]
            print("Prediction = ",i2t[predict], "  img id = ", img_id)
            print("total    score =   %.6f" % score.data[0][predict])
            print("question score =   %.6f" % score_q.data[0][predict])
            print("image    score =   %.6f" % score_i.data[0][predict])

            score.data[0][predict] = -1E6

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
        print("top 6 image")
        for i in range(6):
            print (i+1, "=", words[i], " -> ", score[i])
