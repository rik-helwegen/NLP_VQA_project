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
import json
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import re

# TODO: something takes very long now, would it be the stemmer? We could temporarily shut it down while testing.

# TODO: save the network weights to make testing and continuing training easier

# TODO: plot the loss function, to see what learning rate is suitable.

torch.manual_seed(42)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

# stemmer used to combine 'have, having' into one stem
stemmer = SnowballStemmer("english")
cnt = Counter()

# count word occurence for questions data set
def preprocess_dataset_questions(filename):
    with open(filename, 'r') as file:
        qdata = json.load(file)
        for question in range(len(qdata['questions'])):
            words = qdata['questions'][question]['question']
            words = words.split()
            words[len(words)-1] = words[len(words)-1].replace("?", "")
            words = [re.sub('[^0-9a-zA-Z]+', '', x) for x in words]
            words = [stemmer.stem(x) for x in words]
            for word in words:
                cnt[w2i[word]] += 1

# make a list of all words occurring 'freq' times
def makeList(freq):
    removeWords = []
    for combo in cnt.most_common():
        if (combo[1] <= freq):
            removeWords.append(combo[0])
    return removeWords

# TODO: Question: how is this possible before def preprocess_dataset_questions has been called for?
# make list with word occurence of 1 (change 1 to different number for other results)
remove = makeList(1)

# to read in the questions
def read_dataset_questions(filename):
    with open(filename, 'r') as file:
        qdata = json.load(file)
        for question in range(len(qdata['questions'])):
            words = qdata['questions'][question]['question']
            img_id = qdata['questions'][question]['image_id']
            # preprocessing the words
            words = words.split()
            words[len(words)-1] = words[len(words)-1].replace("?", "")
            words = [re.sub('[^0-9a-zA-Z]+', '', x) for x in words]
            words = [stemmer.stem(x) for x in words]
            # data structure: [ [img_id, question] ] for every question, and removes unary occurences.
            yield ([img_id, [w2i[x] for x in words if x not in remove]])

# to read in the answers
def read_dataset_answers(filename):
    with open(filename, 'r') as file:
        adata = json.load(file)
        for annotation in range(len(adata['annotations'])):
            answer = adata['annotations'][annotation]['multiple_choice_answer']
            answer = re.sub('[^0-9a-zA-Z]+', '', answer)
            answer = stemmer.stem(answer)
            yield (t2i[answer])

# count words first
preprocess_dataset_questions('data/vqa_questions_train.txt')
# loading the word data
x_train = list(read_dataset_questions('data/vqa_questions_train.txt'))
y_train = list(read_dataset_answers('data/vqa_annotatons_train.txt'))
w2i = defaultdict(lambda: UNK, w2i)
x_test = list(read_dataset_questions('data/vqa_questions_test.txt'))
y_test = list(read_dataset_answers('data/vqa_annotatons_test.txt'))

# loading the image data
path_to_h5_file   = 'data/VQA_image_features.h5'
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
training_size = len(x_train)
test_size = len(x_test)
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
        # concatanate embeddings with the image features
        embedding_output = self.embedding_output(embeds)
        img_output = self.img_output(image_input)
        addition = torch.add(embedding_output, img_output)

        return addition


model = CBOW(nwords, 2000, nfeatures, ntags)
print(model)

# TODO Question: Do we need to evaluate the whole test_data as one batch?
def evaluate(model, test_data):
    """Evaluate a model on a data set."""
    correct = 0.0

    # forward with batch size = 1
    for i in range(len(test_data)):
        question = test_data[i][0]
        answer = test_data[i][1]
        img_id = test_data[i][2]
        image = np.ndarray.tolist(img_features[visual_feat_mapping[str(img_id)]])

        question_tensor = Variable(torch.FloatTensor([question]))
        image_tensor = Variable(torch.FloatTensor([image]))

        scores = model(question_tensor, image_tensor)
        predict = scores.data.numpy().argmax(axis=1)[0]
        if predict == answer:
            correct += 1

    return correct, len(x_test), correct / len(test_data)


optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# different layers must use different learning rates
# optimizer = optim.Adam([
#     {'params': model.embedding.parameters(), 'lr': 1e-1}
#     , {'params': model.only_embedding.parameters(), 'lr': 1e-3}
# ])
batch_size = 50

# TODO check if it doesn't learn to always output 'yes', because it outputs a lot of 17's
# TODO loss doesn't go down every iteration, but in general it does.
# with only words it learns super quickly??, with batchsize 1; the acc goes to 20 % after 10 questions.
for ITER in range(50):
    train_loss = 0.0
    start = time.time()

    # make a batch by sampling randomly and split up in input and targets
    batch = training_data[np.random.choice(training_data.shape[0], batch_size, replace=False), :]
    input_questions = [x[0] for x in batch]
    input_targets = [x[1] for x in batch]
    input_img_ids = [x[2] for x in batch]

    # make a batch of image features by using corresponding img_id, and transforms them to a list (needed for FloatTensor)
    images_input = [ np.ndarray.tolist(img_features[visual_feat_mapping[str(i)]]) for i in input_img_ids]

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

# evaluate
_, _, acc = evaluate(model, test_data)
print("iter %r: test acc=%.6f" % (ITER, acc))