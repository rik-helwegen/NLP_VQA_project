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
    with open("data_processed/x_test_type.pkl", "rb") as fp:   #Pickling
        x_test = pickle.load(fp)
    with open("data_processed/y_test_type.pkl", "rb") as fp:   #Pickling
        y_test = pickle.load(fp)

    #size of data used
    # test_size = len(x_test)
    test_size = 2962

    # use size for shorter time
    x_test = x_test[:test_size]
    img_ids_test = [x[0] for x in x_test]
    questions_test = [x[1] for x in x_test]
    print(len(questions_test))

    # test
    answers_test = y_test[:test_size]
    question_type = [x[0] for x in y_test]
    answer_type = [x[1] for x in y_test]
    answers = [x[2] for x in y_test]

    # encodes to one hot vector
    questions_test = one_hot_encoding(questions_test)
    print(questions_test)

    # combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
    test_data = [[questions_test[i], answers[i], img_ids_test[i], question_type[i], answer_type[i]] for i in range(len(questions_test))]
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
print(len(test_data))

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
model.load_state_dict(torch.load('model/testing_model.pkl'))

# create zero vectors to save progress
# learning_train = np.zeros([epochs, 2])
# learning_validation = np.zeros([int(math.floor((epochs-1)/validation_update))+1, 3])


"""Evaluate a model on a data set."""
test_loss = 0.0
correct = 0.0
correctFive = 0.0
np.random.shuffle(test_data)

correct_answers = []
predict_answers = []
# forward with batch size = 1
for i in range(len(test_data)):

    question = test_data[i][0]
    answer = test_data[i][1]

    img_id = test_data[i][2]
    image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

    question_tensor = Variable(torch.FloatTensor([question]))
    image_features_tensor = Variable(torch.FloatTensor([image]))

    score, score_q, score_i = model(question_tensor, image_features_tensor)

    loss = nn.CrossEntropyLoss()

    target = Variable(torch.LongTensor([answer]))

    output = loss(score, target)

    test_loss += output.data[0]

    # measure accuracy of prediction



    predict = score.data.numpy().argmax(axis=1)[0]
    predict_answers.append(predict)
    if predict == answer:
        correct += 1
        correct_answers.append(answer)

        top5 = score.clone()
        fiveAnswers = []

    for iterate in range(5):
        predict = top5.data.numpy().argmax(axis=1)[0]
        top5.data[0][predict] = 0
        fiveAnswers.append(predict)

    if answer in fiveAnswers:
        correctFive += 1

print("testing on #", len(test_data))
print("# in top5:", correctFive)
print("# in top1:", correct)
print("top5 accuracy=", (correctFive/len(test_data))*100)
print("top1 accuracy=", (correct/len(test_data))*100)
avg_test_loss = test_loss/len(test_data)
print(avg_test_loss, len(set(correct_answers)), len(set(predict_answers)))
