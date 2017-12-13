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

training_size = 100000000
# test_size = len(x_test)
test_size = 100
# validation test_size
validation_size = 200000000

x_train = x_train[:training_size]
x_test = x_test[:test_size]
x_validation = x_validation[:validation_size]

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
# encodes to one hot vector
questions_test = one_hot_encoding(questions_test)

# validation
answers_validation = y_validation
img_ids_validation = [x[0] for x in x_validation]
questions_validation = [x[1] for x in x_validation]
# encodes to one hot vector
questions_validation = one_hot_encoding(questions_validation)

# combine questions and answers to [ [question[i], answer[i]], img_id[i]] (for every i)
training_data = [[questions_train[i], answers_train[i], img_ids_train[i]] for i in range(len(questions_train))]
training_data = np.asarray(training_data)

test_data = [[questions_test[i], answers_test[i], img_ids_test[i]] for i in range(len(questions_test))]
test_data = np.asarray(test_data)

validation_data = [[questions_validation[i], answers_validation[i], img_ids_validation[i]] for i in range(len(questions_validation))]
validation_data = np.asarray(validation_data)


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


model = CBOW(nwords, 164, nfeatures, ntags)

print(model)


def evaluate(model, data):
    """Evaluate a model on a data set."""
    test_loss = 0.0
    correct = 0.0
    np.random.shuffle(data)

    correct_answers = []
    predict_answers = []
    # forward with batch size = 1
    for i in range(len(data)):

        question = data[i][0]
        answer = data[i][1]
        img_id = data[i][2]
        image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

        question_tensor = Variable(torch.FloatTensor([question]))
        image_features_tensor = Variable(torch.FloatTensor([image]))
        scores = model(question_tensor, image_features_tensor)
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


optimizer = optim.Adam(model.parameters(), lr=0.0001)  #, weight_decay=0.00001)
# different layers must use different learning rates
# optimizer = optim.Adam([
#     {'params': model.embedding.parameters(), 'lr': 1e-1}
#     , {'params': model.only_embedding.parameters(), 'lr': 1e-3}
# ])

# TODO look into batch normalisation
# minibatch_size = 35

# TODO check if it doesn't learn to always output 'yes', because it outputs a lot of 17's
# TODO loss doesn't go down every iteration, but in general it does.
# with only words it learns super quickly??, with batchsize 1; the acc goes to 20 % after 10 questions.

# Number of epochs


epochs = 20
# # after which number of epochs we want a evaluation:
validation_update = 1
# create zero vectors to save progress
learning_train = np.zeros([epochs, 2])
learning_validation = np.zeros([int(math.floor((epochs-1)/validation_update))+1, 3])
# learning_validation = np.zeros([int(math.floor((epochs-1)))+1, 3])

print('initial loss')
acc, avg_loss, predict_answers, correct_answers = evaluate(model, validation_data)
print("iter %r: validation loss/sent %.6f, accuracy=%.6f" % (0, avg_loss, acc))

# set parameters for hyper optimization

# Jeroen Run
# LR_list = [0.001]
# Rik run:
LR_list = [0.00001]
# Sierk run:
# LR_list = [0.001]

batch_list = [32, 64, 128]
WD_list = [0]

for WD in WD_list:
    for LR in LR_list:
        for minibatch_size in batch_list:

            model = CBOW(nwords, 164, nfeatures, ntags)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            print('learning rate: %f, weight decay: %f,\n batch size: %i' % (LR, WD, minibatch_size))
            for ITER in range(epochs):
                train_loss = 0.0
                start = time.time()
                batches_count = 0
                # split up data in mini-batches
                for i in range(0, training_data.shape[0], minibatch_size):
                    batches_count += 1
                    np.random.shuffle(training_data)
                    batch = training_data[i:i + minibatch_size]
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

            np.random.shuffle(validation_data)
            valid_data = validation_data  #[:30]
            predictions = []

            for i in range(len(valid_data)):
                question = valid_data[i][0]
                answer = valid_data[i][1]
                img_id = valid_data[i][2]
                image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

                eval_question_tensor = Variable(torch.FloatTensor([question]))
                eval_image_tensor = Variable(torch.FloatTensor([image]))

                eval_scores = model(eval_question_tensor, eval_image_tensor)
                eval_target = Variable(torch.LongTensor([answer]))
                predict = eval_scores.data.numpy().argmax(axis=1)[0]
                # if predict == answer:
                #     correct += 1
                predictions.append(predict)

            # print("predictions =", predictions)

            # get lowest validation loss:
            LL = min(learning_validation[:,1])

            # create plot to save
            title = 'learning rate: %f, weight decay: %f,\nlowest valid. loss: %f, batch size: %i' % (LR, WD, LL, minibatch_size)
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(learning_train[:, 0], learning_train[:, 1], label='Average-train-loss')
            axarr[0].legend()
            axarr[1].plot(learning_validation[:, 0], learning_validation[:, 1], label='Average-validation-loss')
            axarr[1].legend()
            axarr[2].plot(learning_validation[:, 0], learning_validation[:, 2], 'r-', label='Validation Accuracy')
            axarr[2].legend()
            axarr[2].set_xlabel('Iterations')
            plt.suptitle(title)
            # plt.show()
            path = './hyper_parameter_tuning/' + 'LR_%.8f-WD_%.8f-LL%.8f-batch_%i' % (LR, WD, LL, minibatch_size)
            f.savefig(path + '.png',  bbox_inches='tight')

            # save data
            np.save(path + '_valid.npy', learning_validation)
            np.save(path + '_train.npy', learning_train)
            plt.close('all')
