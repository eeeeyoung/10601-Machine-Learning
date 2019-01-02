import sys
import numpy as np

syslist = sys.argv[1:]
testwords = open(syslist[0], "r")
index_to_word = open(syslist[1], "r")
index_to_tag = open(syslist[2], "r")
hmmprior_file = open(syslist[3], "r")
hmmemit_file = open(syslist[4], "r")
hmmtrans_file = open(syslist[5], "r")
predicted_file = open(syslist[6], "w")
metric_file = open(syslist[7], "w")


index_word = {}
index_tag = {}

counter1 = 0
for line in index_to_word:
    word = line.strip()
    index_word[word] = counter1
    counter1 += 1

index_list = []
counter2 = 0
for line in index_to_tag:
    tag = line.strip()
    index_list.append(tag)
    index_tag[tag] = counter2
    counter2 += 1

# Calculating initial(prior) probabilities
hmmprior = np.zeros(len(index_tag))
i = 0
for line in hmmprior_file:
    line_ = line.strip()
    hmmprior[i] = float(line_)
    i += 1


hmmemit = np.zeros((len(index_tag), len(index_word)))

i = 0
for line in hmmemit_file:
    line_ = line.strip().split(" ")
    for j in range(len(line_)):
        hmmemit[i][j] = float(line_[j])
    i += 1


hmmtrans = np.zeros((len(index_tag), len(index_tag)))
i = 0
for line in hmmtrans_file:
    line_ = line.strip().split(" ")
    for j in range(len(index_tag)):
        hmmtrans[i][j] = float(line_[j])
    i += 1

# For reference: (TRAINING) F-B algorithm uses hmmprior, hmmtrans, hmmemit
#                                              As Pi,       a,        b

# Importing TEST data and labels

global predictions_tag
predictions_tag = []
global Log_Like
Log_Like = []


# Forward-Backward Algorithm
def FB(sequence):

    # Initializing
    alpha = np.zeros((len(index_tag), len(sequence)))
    alpha_nor = np.zeros((len(index_tag), len(sequence)))
    beta = np.zeros((len(index_tag), len(sequence)))
    beta_nor = np.zeros((len(index_tag), len(sequence)))

    # Forward
    # Initialize alpha1:

    word0 = sequence[0]
    for i in range(len(index_tag)):
        alpha[i][0] = hmmprior[i]*hmmemit[i][word0]
    for i in range(len(index_tag)):
        # Filling in the normalized version of alpha, by dividing by the sum
        alpha_nor[i][0] = alpha[i][0]/np.sum(alpha[:, 0])

    for i1 in range(1, len(sequence)):
        wordx = sequence[i1]
        for j in range(len(index_tag)):
            sum_list = []
            for k in range(len(index_tag)):
                sum_list.append(alpha_nor[k][i1-1]*hmmtrans[k][j])
            alpha[j][i1] = (hmmemit[j][wordx])*sum(sum_list)
        for j in range(len(index_tag)):
            alpha_nor[j][i1] = alpha[j][i1]/np.sum(alpha[:, i1])

    sum_alpha_T = 0
    for i in range(0, len(index_tag)):
        sum_alpha_T += alpha[i][-1]

    Log_Like.append(np.log(sum_alpha_T))

    # Backward
    # Initialize beta

    for i in range(len(index_tag)):
        beta[i][len(sequence)-1] = 1

    for t in range(len(sequence)-2, -1, -1):
        wordx = sequence[t+1]
        for j in range(len(index_tag)):
            sum_list = []
            # Using Posterior Decoding
            for k in range(len(index_tag)):
                sum_list.append((hmmemit[k][wordx])*(beta[k][t+1])*hmmtrans[j][k])
            beta[j][t] = sum(sum_list)

    # Generating the prediction matrix
    pred = alpha*beta
    predicted_tag = []
    for i in range(len(sequence)):
        # Defining the prediction tag to the most probable prediction
        predicted_tag.append(np.argmax(pred[:, i], axis=0))

    predictions_tag.append(predicted_tag)


test_words = []
test_tags = []
actual_words_output = []

for line in testwords:
    line_ = line.strip().split(" ")

    words = []
    tags = []
    actual_word = []

    for ele in line_:
        word_split1 = ele.split("_")
        word_pos = index_word[word_split1[0]]
        tag_pos = index_tag[word_split1[1]]
        words.append(word_pos)
        tags.append(tag_pos)
        actual_word.append(word_split1[0])

    test_words.append(words)
    test_tags.append(tags)
    actual_words_output.append(actual_word)


def loop():
    for i in range(len(test_words)):
        FB(test_words[i])


loop()


total_word_count = 0
correct_ones = 0
prediction_output = []
for i in range(len(test_words)):
    list__ = []
    for j in range(len(test_words[i])):
        list__.append(str(actual_words_output[i][j])+"_"+str(index_list[predictions_tag[i][j]]))
        total_word_count += 1
        if str(predictions_tag[i][j]) == str(test_tags[i][j]):
            correct_ones += 1
        else:
            pass
    prediction_output.append(list__)


accuracy = correct_ones/total_word_count

Avg_Log_Like = (1/len(test_words))*np.sum(Log_Like)

metric_file.write("Average Log-Likelihood: "+str(Avg_Log_Like)+"\n"+"Accuracy: "+str(accuracy))

for i in range(len(test_words)):
    predicted_file.write(" ".join(ele for ele in prediction_output[i])+"\n")