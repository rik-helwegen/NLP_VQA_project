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

training_size = 10000
# test_size = len(x_test)
test_size = 100
# validation test_size
validation_size = 1000

x_train = x_train[:training_size]
x_test = x_test[:test_size]
x_validation = x_validation[:validation_size]

# for ease, separate the data
# training
answers_train = y_train[:training_size]
img_ids_train = [x[0] for x in x_train]
questions_train = [x[1] for x in x_train]

# make every question evenly long
question_train_equal=[]
length_longest_question = len(max(questions_train, key = len))
for question in questions_train:
    length = len(question)
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

# validation
answers_validation = y_validation
img_ids_validation = [x[0] for x in x_validation]
questions_validation = [x[1] for x in x_validation]

# combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
training_data = [[questions_train[i], answers_train[i], img_ids_train[i]] for i in range(len(questions_train))]
training_data = np.asarray(training_data)

test_data = [[questions_test[i], answers_test[i], img_ids_test[i]] for i in range(len(questions_test))]
test_data = np.asarray(test_data)

validation_data = [[questions_validation[i], answers_validation[i], img_ids_validation[i]] for i in range(len(questions_validation))]
validation_data = np.asarray(validation_data)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, image_features_dim, output_dim, hidden_state_size):
        super(RNN, self).__init__()

        self.ReLu = nn.ReLU().cuda()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=nwords-1).cuda()
        self.output = nn.Linear(embedding_dim + image_features_dim + hidden_state_size, output_dim).cuda()
        self.hidden_in = nn.Linear(embedding_dim + image_features_dim + hidden_state_size, hidden_state_size).cuda()
        self.linear1 = nn.Linear(output_dim, output_dim).cuda()

    def forward(self, words_input, image_input, hidden_state_input):

        embeds = self.embedding(words_input)
        combined = torch.cat((embeds, image_input, hidden_state_input), 1)
        combined = self.ReLu(combined)
        hidden_state_in = self.hidden_in(combined)
        hidden_state_out = self.ReLu(hidden_state_in)

        output_out = self.output(combined)
        output_out = self.linear1(output_out)
        return output_out, hidden_state_out


# size of hidden state
n_hidden = 150

model = RNN(nwords, 164, nfeatures, ntags, n_hidden)

print(model)


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
        image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

        hidden_state = np.ndarray.tolist(np.zeros((1, n_hidden)))
        hidden_state = Variable(torch.cuda.FloatTensor(hidden_state))
        for j in range(len(question)):
            input_j_word = question[j]
            # forward pass
            question_tensor = Variable(torch.cuda.LongTensor([input_j_word]))
            image_features_tensor = Variable(torch.cuda.FloatTensor([image]))
            scores, hidden_state = model(question_tensor, image_features_tensor, hidden_state)
            last_output = scores

        loss = nn.CrossEntropyLoss()
        target = Variable(torch.cuda.LongTensor([answer]))

        output = loss(last_output, target)
        test_loss += output.data[0]

        # measure accuracy of prediction
        predict = last_output.data.cpu().numpy().argmax(axis=1)[0]
        predict_answers.append(predict)
        if predict == answer:
            correct += 1
            correct_answers.append(answer)

    accuracy = correct / len(data) * 100
    avg_test_loss = test_loss/len(data)
    return accuracy, avg_test_loss, len(set(correct_answers)), len(set(predict_answers))

# different layers must use different learning rates
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)
optimizer = optim.SGD([
    {'params': model.embedding.parameters(), 'lr': 0.001},
    {'params': model.output.parameters(), 'lr': 0.00005},
    {'params': model.linear1.parameters(), 'lr': 0.00005}])

# met [0.001, 0.0001] wordt hij eerste 3 iteraties beter, daarna overfit
# met [0.0001, 0,00001] meteen overfit
# met [0.0001, 0.000001] Goede start (5 unique correct, 83 answers) maar dan converged hij naar (2,2), vanaf iter 7 lijkt hij te gaan overfitten (niet gewacht)
# met [0.0001, 0.0000001] (waarom niet) begint met: (1,227) (1,211) (1,202)
# met [0.0001, 0.0000005] (idee: combinatie tussen diversiteit en strijdkracht uit vorige twee) (2,172) (3,106) (3,47) (2,16) diversiteit gaat te snel verloren
# met [0.001, 0.0000005] (idee: meer strijdkracht)(2, 146)(3, 108)(3,43)
# met [0.01, 0.0000005] (1, 171) (4, 105) (3,40) (2,19)
# met [0.0001, 0.0000005] (1, 171) (4, 105) (3,40) (2,19)
# met [0.000001, 0.00000005] (1, 229) (1, 219) (1,207) ..
# met SGD [0.000001, 0.00000005] (1, 235) (1, 235)
# met SGD [0.00001, 0.0000005] (1, 235) (1, 235)
# met SGD [0.00001, 0.000005] (iets meer beslis power) (1,235)(1,235) echt, niet per ongeluk kopie
# met SGD [0.0001, 0.00005] (meer beslis power) (2,2)
# met SGD [0.0001, 0.000005] (iets minder beslis power) (1, 235)(3,226)(2,113)(2,11)(2,2)(2,2)(2,2)(2,2)(2,2)(2,2)
# met SGD [0.001, 0.0005] uit dat loakel optimum komen (2,2)(2,2)
# met SGD [0.1, 0.0005] uit dat loakel optimum komen (2,2)(2,2)
# met SGD [0.001, 0.00005] uit dat loakel optimum komen (2,2)
# met SGD extra hidden layer [0.001, 0.00005] (2,2) (2,2)

minibatch_size = 10
# Number of epochs
epochs = 40
# # after which number of epochs we want a evaluation:
validation_update = 1
# create zero vectors to save progress
learning_train = np.zeros([epochs, 2])
learning_validation = np.zeros([int(math.floor((epochs-1)/validation_update))+1, 3])
# learning_validation = np.zeros([int(math.floor((epochs-1)))+1, 3])

print('initial loss')
acc, avg_loss, predict_answers, correct_answers = evaluate(model, validation_data)
print("iter %r: validation loss/sent %.6f, accuracy=%.6f" % (0, avg_loss, acc))

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
        input_questions = np.asarray(input_questions)

        # make a batch of image features by using corresponding img_id, and transforms them to a list (needed for FloatTensor)
        images_input = [np.ndarray.tolist(img_features[visual_feat_mapping[str(i)]]) for i in input_img_ids]

        hidden_state = np.ndarray.tolist(np.zeros((len(input_questions), n_hidden)))
        hidden_state = Variable(torch.cuda.FloatTensor(hidden_state))
        for j in range(len(input_questions[0])):
            input_j_words = np.ndarray.tolist(input_questions[:, j])

            # forward pass
            question_tensor = Variable(torch.cuda.LongTensor(input_j_words))
            image_features_tensor = Variable(torch.cuda.FloatTensor(images_input))
            scores, hidden_state = model(question_tensor, image_features_tensor, hidden_state)
            last_output = scores

        loss = nn.CrossEntropyLoss()
        target = Variable(torch.cuda.LongTensor(input_targets))

        output = loss(last_output, target)
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
        acc, avg_loss, correct_answers, predict_answers = evaluate(model, validation_data)
        print("iter %r: validation loss/sent %.6f, accuracy=%.6f" % (ITER, avg_loss, acc))
        learning_validation[ITER, :] = [ITER, avg_loss, acc]
        print("Unique correct answers", correct_answers)
        print("Unique predict answers", predict_answers)
    path = './hyper_parameter_tuning/' + 'RNN_ITER_%i' % (ITER) + '.pt'
    torch.save(model.state_dict(), path)
    path = './hyper_parameter_tuning/' + 'RNN_ITER_%i' % (ITER)
    np.save(path + '_valid.npy', learning_validation)
    np.save(path + '_train.npy', learning_train)

plt.close('all')
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(learning_train[:, 0], learning_train[:, 1], label='Average-train-loss')
axarr[0].legend()
axarr[1].plot(learning_validation[:, 0], learning_validation[:, 1], label='Average-validation-loss')
axarr[1].legend()
axarr[2].plot(learning_validation[:, 0], learning_validation[:, 2], 'r-', label='Accuracy')
axarr[2].legend()
axarr[2].set_xlabel('Iterations')
plt.show()

path = './hyper_parameter_tuning/RNN'
f.savefig(path + '.png',  bbox_inches='tight')

# save data
np.save(path + '_valid.npy', learning_validation)
np.save(path + '_train.npy', learning_train)
plt.close('all')
