import sys
import numpy as np

list = sys.argv[1:]
trainwords = open(list[0], "r")
index_to_word = open(list[1], "r")
index_to_tag = open(list[2], "r")
hmmprior_file = open(list[3], "w")
hmmemit_file = open(list[4], "w")
hmmtrans_file = open(list[5], "w")

# Initialize dictionaries for receiving words and tags
index_word = {}
index_tag = {}

counter1 = 0
for line in index_to_word:
    word = line.strip()
    index_word[word] = counter1
    counter1 += 1

counter2 = 0
for line in index_to_tag:
    tag = line.strip()
    index_tag[tag] = counter2
    counter2 += 1

# Calculating initial probabilities
hmmemit_count = np.ones((len(index_tag), len(index_word)))
init_prob_count = np.zeros(len(index_tag))
hmmtrans_count = np.ones((len(index_tag),len(index_tag)))

for line in trainwords:
    line_ = line.strip().split(" ")

    # hmmprior

    first_word = line_[0]
    word_split = first_word.split("_")
    first_tag = word_split[1]
    tag_pos = index_tag[first_tag]
    init_prob_count[int(tag_pos)] += 1

    # hmmemit
    states = []

    for ele in line_:
        word_split1 = ele.split("_")
        word_pos = index_word[word_split1[0]]
        tag_pos = index_tag[word_split1[1]]
        hmmemit_count[tag_pos][word_pos] += 1
        states.append(tag_pos)

    # hmmtrans
    for k in range(len(states)-1):
        a = states[k]
        b = states[k + 1]
        hmmtrans_count[a][b] += 1



hmmemit = np.zeros((len(index_tag), len(index_word)))
for i in range(len(index_tag)):
    for j in range(len(index_word)):
        hmmemit[i][j] = hmmemit_count[i][j]/np.sum(hmmemit_count[i])


init_prob_count_ = init_prob_count + 1
hmmprior = np.zeros(len(index_tag))
for i in range(len(index_tag)):
    hmmprior[i] = init_prob_count_[i]/sum(init_prob_count_)

hmmtrans = np.zeros((len(index_tag), len(index_tag)))
for i in range(len(index_tag)):
    for j in range(len(index_tag)):
        hmmtrans[i][j] = hmmtrans_count[i][j]/np.sum(hmmtrans_count[i])

# Writing Files:


hmmtrans_list = []
for i in range(len(index_tag)):
    list_ = []
    for j in range(len(index_tag)):
        list_.append(str(hmmtrans[i][j]))
    hmmtrans_list.append(list_)

for j in range(len(index_tag)):
    hmmtrans_file.write(" ".join(ele for ele in hmmtrans_list[j])+"\n")


hmmemit_list = []
for i in range(len(index_tag)):
    list_ = []
    for j in range(len(index_word)):
        list_.append(str(hmmemit[i][j]))
    hmmemit_list.append(list_)

for j in range(len(index_tag)):
    hmmemit_file.write(" ".join(ele for ele in hmmemit_list[j])+"\n")


hmmprior_list = []
for i in range(len(index_tag)):
    hmmprior_list.append(str(hmmprior[i]))

for j in range(len(index_tag)):
    hmmprior_file.write(str(hmmprior_list[j])+"\n")