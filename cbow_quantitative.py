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

    img_ids_test   = [x[0] for x in x_test]
    questions_test = [x[1] for x in x_test]
    answers        = [x for x in y_test]
    answer_type    = [x for x in y_test_type]

    # encodes to one hot vector
    questions_test = one_hot_encoding(questions_test)

    # combine question, answer, image id and answer type to one array
    test_data = [[questions_test[i], answers[i], img_ids_test[i], answer_type[i]] for i in range(len(questions_test))]
    test_data = np.asarray(test_data)

    #return test data
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

# create path to image features file
path_to_h5_file = 'data/VQA_image_features.h5'
path_to_json_file = 'data/VQA_img_features2id.json'

# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

# load mapping file
with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

# load vocabularies from data
with open("data_processed/w2i.pkl", "rb") as fp:   #Pickling
    w2i = pickle.load(fp)
with open("data_processed/t2i.pkl", "rb") as fp:   #Pickling
    t2i = pickle.load(fp)

#length vocabulary of questions, answers and image features
nwords = len(w2i)
ntags = len(t2i)
nfeatures = len(img_features[0])

# load test data set
test_data = dataLoader()

# set parameters for network
vocwords = nwords
voctags = ntags
vocfeatures = nfeatures

#create model
model = CBOW(vocwords, 164, vocfeatures, voctags)

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
# correct amount of answers in top 1
correct = [0.0, 0.0, 0.0, 0.0]
# correct amount of answers in top 5
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
    question =    test_data[i][0]
    answer   =    test_data[i][1]
    img_id   =    test_data[i][2]
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

    # check prediction with highest score from output network
    predict = score.data.numpy().argmax(axis=1)[0]

    # get an list with tuples of index and
    predict_sorted = []
    index = 0
    for pre in score.data[0]:
        predict_sorted.append([index,pre])
        index +=1
    predict_sorted = sorted(predict_sorted, key=lambda score: score[1], reverse=True)
    predict_sorted = np.asarray(predict_sorted)[:,0].tolist()

    #Add one to position
    position_array[predict_sorted.index(answer)] +=1

    #if type corresponds, add one to correct, also add one to corresponding position
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
        #if type corresponds, add one and corresponding position array
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
        #find to which type the correct answer belongs, add one to correct
        if answer_type == 'yes/no':
            correct_five[1] += 1
        elif answer_type == 'other':
            correct_five[2] += 1
        elif answer_type == 'number':
            correct_five[3] += 1

#create plot with 4 subplots
f, axarr = plt.subplots(2,2)

#plot 4 subplots with each plot containing a position array
#plot 0:40 positions for better overview
axarr[0,0].plot(position_array[0:40],label ="total")
axarr[0,1].plot(position_arrayYN[0:40], label = "Yes/No")
axarr[1,0].plot(position_arrayNU[0:40], label = "Number")
axarr[1,1].plot(position_arrayOT[0:40], label = "Other")

#set legends for plot
axarr[0,0].legend()
axarr[0,1].legend()
axarr[1,0].legend()
axarr[1,1].legend()

#subtitle of
plt.suptitle("Distribution of answered ranked on position in output")

#set labels on axis
axarr[0,0].set_ylabel("Amount on position")
axarr[1,0].set_ylabel("Amount on position")
axarr[1,0].set_xlabel("Position from highest rank")
axarr[1,1].set_xlabel("Position from highest rank")

#set x axis limit to 1000 for better overview
axarr[0,0].set_ylim([0, 1000])
axarr[0,1].set_ylim([0, 1000])
axarr[1,1].set_ylim([0, 1000])
axarr[1,0].set_ylim([0, 1000])


if(which_model == "loss"):
    #save plot of quantitative_cbow data
    f.savefig('data_gathered/quantitative_cbow_loss.png')
    #save array with data of positions
    np.save('data_gathered/bow_low_loss_total.npy',  position_array)
    np.save('data_gathered/bow_low_loss_number.npy', position_arrayNU)
    np.save('data_gathered/bow_low_loss_yes_no.npy', position_arrayYN)
    np.save('data_gathered/bow_low_loss_other.npy',  position_arrayOT)
else:
    #save plot of quantitative_cbow data
    f.savefig('data_gathered/quantitative_cbow_acc.png')
    #save array with data of positions
    np.save('data_gathered/bow_high_acc_total.npy',  position_array)
    np.save('data_gathered/bow_high_acc_number.npy', position_arrayNU)
    np.save('data_gathered/bow_high_acc_yes_no.npy', position_arrayYN)
    np.save('data_gathered/bow_high_acc_other.npy',  position_arrayOT)

#calculate average test loss on test data set
avg_test_loss = test_loss/len(test_data)

#print amount of data and average test loss
print("testing on #", len(test_data), "  avg test loss =", avg_test_loss)

#Print statements for top 5 accuracy of total
print("# in top5:", correct_five[0], "    # in top1:", correct[0])
print("top1 accuracy total=", (correct[0]/len(test_data))*100)
print("top5 accuracy total =", (correct_five[0]/len(test_data))*100)

#Print statements for top 5 accuracy of yes/no
print("# in top5:", correct_five[1], "    # in top1:", correct[1])
print("top1 accuracy yes/no=", (correct[1]/total_yesno)*100)
print("top5 accuracy yes/no =", (correct_five[1]/total_yesno)*100)
#Print statements for top 5 accuracy on subset
print("top1 accuracy yes/no=", (correct[1]/len(test_data))*100)
print("top5 accuracy yes/no =", (correct_five[1]/len(test_data))*100)

#Print statements for top 5 accuracy of other
print("# in top5:", correct_five[2], "    # in top1:", correct[2])
print("top1 accuracy other=", (correct[2]/total_other)*100)
print("top5 accuracy other=", (correct_five[2]/total_other)*100)
#Print statements for top 5 accuracy on subset
print("top1 accuracy other=", (correct[2]/len(test_data))*100)
print("top5 accuracy other=", (correct_five[2]/len(test_data))*100)

#Print statements for top 5 accuracy of number
print("# in top5:", correct_five[3], "    # in top1:", correct[3])
print("top1 accuracy number=", (correct[3]/total_number)*100)
print("top5 accuracy number=", (correct_five[3]/total_number)*100)
#Print statements for top 5 accuracy on subset
print("top1 accuracy number=", (correct[3]/len(test_data))*100)
print("top5 accuracy number=", (correct_five[3]/len(test_data))*100)
