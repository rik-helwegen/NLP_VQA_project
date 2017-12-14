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
import dill as pickle


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
            words = [x.lower() for x in words]
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
            words = [x.lower() for x in words]
            words = [stemmer.stem(x) for x in words]
            # data structure: [ [img_id, question] ] for every question, and removes unary occurences.
            yield ([img_id, [w2i[x] for x in words if x not in remove]])

# to read in the answers
def read_dataset_answers(filename):
    with open(filename, 'r') as file:
        adata = json.load(file)
        for annotation in range(len(adata['annotations'])):
            answer = adata['annotations'][annotation]['multiple_choice_answer']
            a_type = adata['annotations'][annotation]['answer_type']
            q_type = adata['annotations'][annotation]['question_type']
            answer = re.sub('[^0-9a-zA-Z]+', '', answer)
            answer = stemmer.stem(answer)
            yield (a_type, q_type, t2i[answer])

# count words first
preprocess_dataset_questions('data/vqa_questions_train.txt')

# make list with word occurence of 1 (change 1 to different number for other results)
remove = makeList(1)

x_test_type = list(read_dataset_questions('data/vqa_questions_test.txt'))
y_test_type = list(read_dataset_answers('data/vqa_annotatons_test.txt'))

x_vali_type = list(read_dataset_questions('data/vqa_questions_valid.txt'))
y_vali_type = list(read_dataset_answers('data/vqa_annotatons_valid.txt'))

# dump test set
with open("data_processed/x_test_type.pkl", "wb") as fp:   #Pickling
    pickle.dump(x_test_type, fp)
with open("data_processed/y_test_type.pkl", "wb") as fp:   #Pickling
    pickle.dump(y_test_type, fp)

# dump validation set
with open("data_processed/x_validation_type.pkl", "wb") as fp:   #Pickling
    pickle.dump(x_vali_type, fp)
with open("data_processed/y_validation_type.pkl", "wb") as fp:   #Pickling
    pickle.dump(y_vali_type, fp)
