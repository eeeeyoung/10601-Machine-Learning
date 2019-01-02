import numpy as np
import sys

arg_list = sys.argv[1:]
train_input = open(arg_list[0], "r")
validation_input = open(arg_list[1], "r")
test_input = open(arg_list[2], "r")
dict_input = arg_list[3]
formatted_train_out = open(arg_list[4], "w")
formatted_validation_out = open(arg_list[5], "w")
formatted_test_out = open(arg_list[6], "w")
feature_flag = arg_list[7]

feature_threshold = 4

sys.setrecursionlimit(100000)

# Reading words from dictionary file
word_dict = {}
dict_file = open(dict_input, "r")
for line in dict_file:
    line_ = line.strip().split(" ")
    word_dict[line_[0]] = line_[1]

# For local testing purposes without calling all arguments
"""
train_input = open("train_data.tsv", "r")
validation_input = open("valid_data.tsv", "r")
test_input = open("test_data.tsv", "r")
formatted_train_output = open('formatted_train_output.tsv', "w")
formatted_test_output = open("formatted_test_output.tsv", "w")
formatted_validation_output = open("formatted_validation_output.tsv", "w")
"""


# Model 1 considers every word that appeared in a sentence, no matter how many times it appeared
# As opposed to Model 2, who only considers a word when its appearance in a sentence does not exceed 4 times
def model_1():

    for line in train_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        for ele in review_:
            try:
                # only includes a word when it is not already in word_list
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            output.append("\t"+str(ele)+":1")
        formatted_train_out.write("".join(ele for ele in output)+"\n")

    for line in test_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        for ele in review_:
            try:
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            output.append("\t"+str(ele)+":1")
        formatted_test_out.write("".join(ele for ele in output)+"\n")

    for line in validation_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        for ele in review_:
            try:
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            output.append("\t"+str(ele)+":1")
        formatted_validation_out.write("".join(ele for ele in output)+"\n")


def model_2():

    for line in train_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        review_raw_list = []
        for ele in review_:
            try:
                review_raw_list.append(word_dict[ele])
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            if review_raw_list.count(ele) < 4:
                output.append("\t"+str(ele)+":1")
        formatted_train_out.write("".join(ele for ele in output)+"\n")

    for line in test_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        review_raw_list = []
        for ele in review_:
            try:
                review_raw_list.append(word_dict[ele])
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            if review_raw_list.count(ele) < 4:
                output.append("\t"+str(ele)+":1")
        formatted_test_out.write("".join(ele for ele in output)+"\n")

    for line in validation_input:
        line_ = line.strip().split("\t")
        label = line_[0]
        review = line_[1]
        review_ = review.split(" ")

        word_list = []
        review_raw_list = []
        for ele in review_:
            try:
                review_raw_list.append(word_dict[ele])
                if word_dict[ele] not in word_list:
                    word_list.append(word_dict[ele])
            except KeyError:
                pass

        output=[]
        output.append(label)
        for ele in word_list:
            if review_raw_list.count(ele) < 4:
                output.append("\t"+str(ele)+":1")
        formatted_validation_out.write("".join(ele for ele in output)+"\n")


if int(feature_flag) == 1:
    print("mode = 1")
    model_1()

elif int(feature_flag) == 2:
    print("mode = 2")
    model_2()