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
training_size = 1000
# test_size = len(x_test)
test_size = 100
x_train = x_train[:training_size]
x_test = x_test[:test_size]

# for ease, separate the data
# training
answers_train = y_train[:training_size]
img_ids_train = [x[0] for x in x_train]
questions_train = [x[1] for x in x_train]

# encodes to one hot vector
questions_train = one_hot_encoding(questions_train)

# test
answers_test = y_test[:test_size]
img_ids_test = [x[0] for x in x_test]
questions_test = [x[1] for x in x_test]
questions_test = one_hot_encoding(questions_test)

# combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
training_data = [[questions_train[i], answers_train[i], img_ids_train[i]] for i in range(len(questions_train))]
training_data = np.asarray(training_data)

test_data = [[questions_test[i], answers_test[i], img_ids_test[i]] for i in range(len(questions_test))]
test_data = np.asarray(test_data)


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
        # img_output = self.img_output(image_input)
        # addition = torch.add(embedding_output, img_output)

        return embedding_output


model = CBOW(nwords, 2000, nfeatures, ntags)

print(model)


# # TODO Question: Do we need to evaluate the whole test_data as one batch?
def evaluate(model, test_data):
    """Evaluate a model on a data set."""
    eval_loss = 0.0

    # forward with batch size = 1
    for i in range(len(test_data)):

        question = test_data[i][0]
        answer = test_data[i][1]
        img_id = test_data[i][2]
        image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

        eval_question_tensor = Variable(torch.FloatTensor([question]))
        eval_image_tensor = Variable(torch.FloatTensor([image]))

        eval_scores = model(eval_question_tensor, eval_image_tensor)
        eval_target = Variable(torch.LongTensor([answer]))

        loss = nn.CrossEntropyLoss()

        eval_output = loss(eval_scores, eval_target)
        eval_loss += eval_output.data[0]

    return eval_loss, eval_loss / len(test_data)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
# different layers must use different learning rates
# optimizer = optim.Adam([
#     {'params': model.embedding.parameters(), 'lr': 1e-1}
#     , {'params': model.only_embedding.parameters(), 'lr': 1e-3}
# ])

# TODO look into batch normalisation
minibatch_size = 50

# TODO check if it doesn't learn to always output 'yes', because it outputs a lot of 17's
# TODO loss doesn't go down every iteration, but in general it does.
# with only words it learns super quickly??, with batchsize 1; the acc goes to 20 % after 10 questions.

# Number of epochs
epochs = 10
# after which number of epochs we want a evaluation:
test_update = 1
# create zero vectors to save progress
learning_train = np.zeros([epochs, 2])
learning_test = np.zeros([int(math.floor((epochs-1)/test_update))+1, 2])

for ITER in range(epochs):
    train_loss = 0.0
    start = time.time()

    # split up data in mini-batches
    for i in range(0, training_data.shape[0], minibatch_size):
        batch = training_data[i:i + minibatch_size]
        y_train_mini = y_train[i:i + minibatch_size]
        input_questions = [x[0] for x in batch]
        input_targets = [x[1] for x in batch]
        input_img_ids = [x[2] for x in batch]

        # make a batch of image features by using corresponding img_id, and transforms them to a list (needed for FloatTensor)
        images_input = [np.ndarray.tolist(img_features[visual_feat_mapping[str(i)]]) for i in input_img_ids]

        # forward pass
        question_tensor = Variable(torch.FloatTensor(input_questions))
        image_features_tensor = Variable(torch.FloatTensor(images_input))
        scores = model(question_tensor, image_features_tensor)
        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor(input_targets))

        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: train loss/sent %.6f, time=%.2fs" %
          (ITER, train_loss/len(training_data), time.time() - start))

    # evaluate, only evaluate every 'test_update' iterations, to save time
    if ITER % test_update == 0:
        eval_loss, avg_eval_loss = evaluate(model, test_data)
        print("iter %r: test loss=%.6f, average test loss=%.6f" % (ITER, eval_loss, avg_eval_loss))
        learning_test[int(ITER/test_update), :] = [ITER, avg_eval_loss]

    learning_train[ITER, :] = [ITER, train_loss/len(training_data)]


plt.close('all')
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(learning_train[:, 0], learning_train[:, 1], label='Average-train-loss')
axarr[0].legend()
axarr[1].plot(learning_test[:, 0], learning_test[:, 1], label='Average-test-loss')
axarr[1].plot(learning_test[:, 0], learning_test[:, 1], 'ro')
axarr[1].legend()
axarr[1].set_xlabel('Iterations')
plt.show()
