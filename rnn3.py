# coding: utf-8

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

# TODO: save the network weights to make testing and continuing training easier
# TODO: plot the loss function, to see what learning rate is suitable.

torch.manual_seed(42)

# Load preprocessed data from pickle files.
with open("data_processed/x_train.pkl", "rb") as fp:   #Pickling
    x_train = pickle.load(fp)
with open("data_processed/y_train.pkl", "rb") as fp:   #Pickling
    y_train = pickle.load(fp)

with open("data_processed/x_test.pkl", "rb") as fp:   #Pickling
    x_test = pickle.load(fp)
with open("data_processed/y_test.pkl", "rb") as fp:   #Pickling
    y_test = pickle.load(fp)

with open("data_processed/x_validation.pkl", "rb") as fp:   #Pickling
    x_validation = pickle.load(fp)
with open("data_processed/y_validation.pkl", "rb") as fp:   #Pickling
    y_validation = pickle.load(fp)

with open("data_processed/w2i.pkl", "rb") as fp:   #Pickling
    w2i = pickle.load(fp)
with open("data_processed/t2i.pkl", "rb") as fp:   #Pickling
    t2i = pickle.load(fp)

# loading the image data
path_to_h5_file = 'data/VQA_image_features.h5'
path_to_json_file = 'data/VQA_img_features2id.json'

# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

# load mapping file
with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

nwords = len(w2i)
ntags = len(t2i)
nfeatures = len(img_features[0])

print(nwords, ntags, nfeatures)

# encodes one-hot with length of vocab size
def one_hot_encoding(data):
    embeddings = np.zeros((len(data), nwords))
    for i, question in enumerate(data):
        for word in question:
            embeddings[i][word] = 1
    return embeddings

# to speed-up training, max = len(x_train)
# training_size = len(x_train)
training_size = 1000000
# test_size = len(x_test)
test_size = 100000
# validation test_size
validation_size = 100000

x_train = x_train[:training_size]
x_test = x_test[:test_size]
x_validation = x_validation[:validation_size]

x_train = x_train + x_validation
x_train = x_train[:57024]
# for ease, separate the data

# validation

img_ids_validation = [x[0] for x in x_validation]
questions_validation = [x[1] for x in x_validation]
# training
answers_train = y_train
answers_validation = y_validation
answers_train = answers_train + answers_validation
answers_train = answers_train [:57024]
img_ids_train = [x[0] for x in x_train]
questions_train = [x[1] for x in x_train]

# make every question evenly long
question_train_equal=[]
q_length = []
length_longest_question = len(max(questions_train, key = len))+1


for question in questions_train:
    length = len(question)
    q_length.append(length-1)
    for i in range(length_longest_question-length):
        question.append(nwords)
    question_train_equal.append(question)
# print(question_train_equal)
nwords += 1
questions_train = question_train_equal

# test
answers_test = y_test[:test_size]
img_ids_test = [x[0] for x in x_test]
questions_test = [x[1] for x in x_test]


# combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
training_data = [[questions_train[i], answers_train[i], img_ids_train[i], q_length[i]] for i in range(len(questions_train))]
training_data = np.asarray(training_data)

test_data = [[questions_test[i], answers_test[i], img_ids_test[i]] for i in range(len(questions_test))]
test_data = np.asarray(test_data)

# validation_data = [[questions_validation[i], answers_validation[i], img_ids_validation[i]] for i in range(len(questions_validation))]
# validation_data = np.asarray(validation_data)

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
        out = self.fc(image_with_hidden)
        return out



# print(model)

def evaluate(model, data):
    """Evaluate a model on a data set."""
    test_loss = 0.0
    correct = 0.0

    correct_answers = []
    predict_answers = []
    # forward with batch size = 1
    for i in range(len(data)):
        question = data[i][0]
        answer = data[i][1]
        img_id = data[i][2]
        input_q_length = Variable(torch.LongTensor([len(question)-1]))
        image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

        # forward pass
        question_tensor = Variable(torch.LongTensor([question]))
        image_features_tensor = Variable(torch.FloatTensor([image]))
        scores = model(question_tensor, image_features_tensor, 1, hidden_size, input_q_length)
        scores =  scores.view(1,-1)
        # last_output = scores

        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor([answer]))
        output = loss(scores, target)
        test_loss += output.data[0]

        # measure accuracy of prediction
        predict = scores.data.numpy().argmax(axis=1)[0]
        predict_answers.append(predict)
        if predict == answer:
            correct += 1
            correct_answers.append(answer)

    accuracy = correct / len(data) * 100
    avg_test_loss = test_loss/len(data)
    return accuracy, avg_test_loss, len(set(correct_answers)), len(set(predict_answers))


# Number of epochs
epochs = 50
# # after which number of epochs we want a evaluation:
validation_update = 1
# create zero vectors to save progress
learning_train = np.zeros([epochs, 2])
learning_validation = np.zeros([int(math.floor((epochs-1)/validation_update))+1, 3])
# learning_validation = np.zeros([int(math.floor((epochs-1)))+1, 3])

# print('initial loss')
# acc, avg_loss, predict_answers, correct_answers = evaluate(model, validation_data)
# print("iter %r: validation loss/sent %.6f, accuracy=%.6f" % (0, avg_loss, acc))

COMBI = [[0.01, 0.001]]

# COMBI = [[0.001, 0.00001],
#          [0.00001, 0.000001],
#          [0.00001, 0.0000001]]

minibatch_size = 64
hidden_size = 128
embedding_size = 164

for comb in COMBI:
    model = RNN(nwords, embedding_size, nfeatures, hidden_size, 1, ntags)
    optimizer = optim.Adam([
        {'params': model.embed.parameters()     , 'lr': comb[0]},
        {'params': model.fc.parameters()        , 'lr': comb[1]},
        {'params': model.img_output.parameters(), 'lr': comb[1]}])


    for ITER in range(epochs):
        train_loss = 0.0
        start = time.time()
        batches_count = 0
        np.random.shuffle(training_data)
        # split up data in mini-batches
        for i in range(0, training_data.shape[0], minibatch_size):
            batches_count += 1
            batch = training_data[i:i + minibatch_size]
            input_questions = [x[0] for x in batch]
            input_targets = [x[1] for x in batch]
            input_img_ids = [x[2] for x in batch]
            input_q_length = Variable(torch.LongTensor([x[3] for x in batch]))

            input_questions = np.asarray(input_questions)

            # make a batch of image features by using corresponding img_id, and transforms them to a list (needed for FloatTensor)
            images_input = [np.ndarray.tolist(img_features[visual_feat_mapping[str(i)]]) for i in input_img_ids]

            # forward pass
            question_tensor = Variable(torch.LongTensor(input_questions))
            image_features_tensor = Variable(torch.FloatTensor(images_input))

            scores = model(question_tensor, image_features_tensor, len(batch), hidden_size, input_q_length)

            loss = nn.CrossEntropyLoss()

            target = Variable(torch.LongTensor(input_targets))

            scores =  scores.view(minibatch_size,-1)

            output = loss(scores, target)

            train_loss += output.data[0]

            # backward pass
            model.zero_grad()
            output.backward()

            # update weights
            optimizer.step()

        # training progress
        print("iter %r: train loss/sent %.6f, time=%.2fs" %
              (ITER, train_loss/batches_count, time.time() - start))
        learning_train[ITER, :] = [ITER, train_loss/batches_count]

        # testing progress
        if ITER % validation_update == 0:
            acc, avg_loss, correct_answers, predict_answers = evaluate(model, test_data)
            print("iter %r: validation loss/sent %.6f, accuracy=%.6f" % (ITER, avg_loss, acc))
            learning_validation[ITER, :] = [ITER, avg_loss, acc]
            print("Unique correct answers", correct_answers)
            print("Unique predict answers", predict_answers)
        path = './hyper_parameter_tuning/' + 'RNN_ITER_%i' % (ITER) + "COMBI" + str(COMBI.index(comb)) + '.pt'
        torch.save(model.state_dict(), path)


    # save data
    path = './hyper_parameter_tuning/RNN_ITER_%i' % (ITER)
    np.save(path + "COMBI" + str(COMBI.index(comb)) + '_valid.npy', learning_validation)
    np.save(path + "COMBI" + str(COMBI.index(comb)) + '_train.npy', learning_train)

    plt.close('all')
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(learning_train[:, 0], learning_train[:, 1], label='Average-train-loss')
    axarr[0].legend()
    axarr[1].plot(learning_validation[:, 0], learning_validation[:, 1], label='Average-validation-loss')
    axarr[1].legend()
    axarr[2].plot(learning_validation[:, 0], learning_validation[:, 2], 'r-', label='Accuracy')
    axarr[2].legend()
    axarr[2].set_xlabel('Iterations')
    # plt.show()

    path = './hyper_parameter_tuning/RNN'
    f.savefig(path + "COMBI" + str(COMBI.index(comb)) + '.png',  bbox_inches='tight')
    plt.close('all')
