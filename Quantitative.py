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

#length vocabulary of questions, answers and image features
nwords = len(w2i)
ntags = len(t2i)
nfeatures = len(img_features[0])

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

#create model
model = CBOW(vocwords, 164, vocfeatures, voctags)
model.load_state_dict(torch.load('model/35_5percentacc.pt'))

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

    question = test_data[i][0]
    answer = test_data[i][1]
    answer_type = test_data[i][3]

    img_id = test_data[i][2]
    image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

    question_tensor = Variable(torch.FloatTensor([question]))
    image_features_tensor = Variable(torch.FloatTensor([image]))

    score, score_q, score_i = model(question_tensor, image_features_tensor)

    loss = nn.CrossEntropyLoss()

    target = Variable(torch.LongTensor([answer]))

    output = loss(score, target)

    test_loss += output.data[0]



    ##Get top 1 accuracy for all 4##


    # check prediction from output network
    predict = score.data.numpy().argmax(axis=1)[0]

    predict_sorted = []
    index = 0
    for pre in score.data[0]:
        predict_sorted.append([index,pre])
        index +=1

    predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
    predict_sorted = np.asarray(predict_sorted)[:,0].tolist()
    position_array[predict_sorted.index(answer)] +=1

    #if type corresponds, add one to correct, also check position
    if answer_type == 'yes/no':
        total_yesno += 1
        position_arrayYN[predict_sorted.index(answer)] +=1
    elif answer_type == 'other':
        total_other += 1
        position_arrayOT[predict_sorted.index(answer)] +=1
    elif answer_type == 'number':
        total_number += 1
        position_arrayNU[predict_sorted.index(answer)] +=1

    # check if prediction is the same as answer
    if predict == answer:
        #add one to total
        correct[0] += 1
        #if type corresponds, add oneposition_array = np.zeros(voctags) to correct
        if answer_type == 'yes/no':
            correct[1] += 1
        elif answer_type == 'other':
            correct[2] += 1
        elif answer_type == 'number':
            correct[3] += 1

    #get clone of score for top 5 accuracy
    top5 = score.clone()
    #get list with top 5 answers to check if it contains answer
    five_answers = []

    #iterate 5 times and set argmax of every iteration to zero
    for iterate in range(5):
        #get current iteration argmax
        predict = top5.data.numpy().argmax(axis=1)[0]
        #set predict 0 for next iter
        top5.data[0][predict] = 0
        #add prediction to check list
        five_answers.append(predict)

    #check if answer is in top 5
    if answer in five_answers:
        #add one to total correct top 5
        correct_five[0] += 1
        #if type corresponds, add one to correct
        if answer_type == 'yes/no':
            correct_five[1] += 1
        elif answer_type == 'other':
            correct_five[2] += 1
        elif answer_type == 'number':
            correct_five[3] += 1


avg_test_loss = test_loss/len(test_data)
print("avg test loss =", avg_test_loss)

print("testing on #", len(test_data))

print(position_array[0:10])

import matplotlib.pyplot as plt
f, axarr = plt.subplots(2,2)
axarr[0,0].plot(position_array[0:40],label ="total")
axarr[0,0].legend()
axarr[0,1].plot(position_arrayYN[0:40], label = "Yes/No")
axarr[0,1].legend()
axarr[1,0].plot(position_arrayNU[0:40], label = "Number")
axarr[1,0].legend()
axarr[1,1].plot(position_arrayOT[0:40], label = "Other")
axarr[1,1].legend()

plt.suptitle("Distribution of answered ranked on position in output")

# axarr[0,0].set_xlabel("Position from highest rank")
axarr[0,0].set_ylabel("Amount on position")
axarr[1,0].set_ylabel("Amount on position")
axarr[1,0].set_xlabel("Position from highest rank")
axarr[1,1].set_xlabel("Position from highest rank")
# axarr[1,1].set_ylabel("Amount on position")

# axarr[0,1].set_xlabel("Position from highest rank")
# axarr[0,1].set_ylabel("Amount on position")

axarr[0,0].set_ylim([0, 1000])
axarr[0,1].set_ylim([0, 1000])
axarr[1,1].set_ylim([0, 1000])
axarr[1,0].set_ylim([0, 1000])

f.savefig('quantitatief.png')

# index 0 -> total, index 1 -> yes/no, index2 -> other, index3 -> number
print("# in top5:", correct_five[0], "    # in top1:", correct[0])
print("top1 accuracy total=", (correct[0]/len(test_data))*100)
print("top5 accuracy total =", (correct_five[0]/len(test_data))*100)

print("# in top5:", correct_five[1], "    # in top1:", correct[1])
print("top1 accuracy yes/no=", (correct[1]/total_yesno)*100)
print("top5 accuracy yes/no =", (correct_five[1]/total_yesno)*100)
print("top1 accuracy yes/no=", (correct[1]/len(test_data))*100)
print("top5 accuracy yes/no =", (correct_five[1]/len(test_data))*100)

print("# in top5:", correct_five[2], "    # in top1:", correct[2])
print("top1 accuracy other=", (correct[2]/total_other)*100)
print("top5 accuracy other=", (correct_five[2]/total_other)*100)
print("top1 accuracy other=", (correct[2]/len(test_data))*100)
print("top5 accuracy other=", (correct_five[2]/len(test_data))*100)

print("# in top5:", correct_five[3], "    # in top1:", correct[3])
print("top1 accuracy number=", (correct[3]/total_number)*100)
print("top5 accuracy number=", (correct_five[3]/total_number)*100)
print("top1 accuracy number=", (correct[3]/len(test_data))*100)
print("top5 accuracy number=", (correct_five[3]/len(test_data))*100)
