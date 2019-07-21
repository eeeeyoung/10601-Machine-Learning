import numpy as np
import sys
import matplotlib.pyplot as plt


"""
np.set_printoptions(threshold=np.inf)
"""
sys.setrecursionlimit(10000)

arg_list = sys.argv[1:]
formatted_train_input = arg_list[0]
formatted_validation_input = arg_list[1]
formatted_test_input = arg_list[2]
dict_input = arg_list[3]
train_output = arg_list[4]
test_output = arg_list[5]
metrics_out = arg_list[6]
num_epoch = arg_list[7]

valid_input = open(formatted_validation_input, "r")
train_input = open(formatted_train_input, "r")
test_input = open(formatted_test_input, "r")

# Reading dict length from dict.txt####
dict_file = open(dict_input, "r").read().strip()
dict_contents = dict_file.split("\n")
dict_size = len(dict_contents)
# Since theta length is dict_length + 1
theta_len = dict_size + 1
# Initialization: Setting intial parameters to all 0s for theta####
theta = np.zeros(theta_len)


# Loading x matrix from the input file
def load_train_input():
    # x_matr is intended to be a list of numpy arrays, storing parameters for x
    global x_matr_train
    x_matr_train = []
    # y is a list storing the label of movie reviews
    global y_train
    y_train = []

    for line in train_input:
        # x_val is a temporary storing list for
        x_val = []
        a = line.strip().split("\t")
        y_train.append(int(a[0]))
        for ele in a[1:]:
            a1 = ele.split(":")  # breaking up the "xxx:1" part bullshit
            b = a1[0]  # keeping only the first part
            x_val.append(b)

        x_i = np.zeros(theta_len)
        x_i[0] = 1
        for ele in x_val:
            k = int(ele) + 1
            x_i[k] = 1
        x_matr_train.append(x_i)


def load_valid_input():
    # x_matr is intended to be a list of numpy arrays, storing parameters for x
    global x_matr_valid
    x_matr_valid = []
    # y is a list storing the label of movie reviews
    global y_valid
    y_valid = []

    for line in valid_input:
        # x_val is a temporary storing list for
        x_val = []
        a = line.strip().split("\t")
        y_valid.append(int(a[0]))
        for ele in a[1:]:
            a1 = ele.split(":")  # breaking up the "xxx:1" part bullshit
            b = a1[0]  # keeping only the first part
            x_val.append(b)

        x_i = np.zeros(theta_len)
        x_i[0] = 1
        for ele in x_val:
            k = int(ele) + 1
            x_i[k] = 1
        x_matr_valid.append(x_i)


def load_test_input():
    # x_matr is intended to be a list of numpy arrays, storing parameters for x
    global x_matr_test
    x_matr_test = []
    # y is a list storing the label of movie reviews
    global y_test
    y_test = []

    for line in test_input:
        # x_val is a temporary storing list for
        x_val = []
        a = line.strip().split("\t")
        y_test.append(a[0])
        for ele in a[1:]:
            a1 = ele.split(":")  # breaking up the "xxx:1" part bullshit
            b = a1[0]  # keeping only the first part
            x_val.append(b)

        x_i = np.zeros(theta_len)
        x_i[0] = 1
        for ele in x_val:
            k = int(ele) + 1
            x_i[k] = 1
        x_matr_test.append(x_i)


load_train_input()
load_valid_input()
load_test_input()

"""
###### sanity checking
print((x_matr_train[1]))
print(y_train)
"""


# function for Stochastic Gradient Descent
def SGD(theta, counter):

    learnR = 0.1
    threshold = len(x_matr_train)

    if counter < threshold -1:

        exp_sth = np.exp(np.dot(theta.transpose(), x_matr_train[counter]))

        new_theta = np.add(theta, learnR*np.dot(x_matr_train[counter], (int(y_train[counter])-(exp_sth/(1+exp_sth)))))

        new_counter = counter + 1

        # Recursively calling SGD
        theta_trans = SGD(new_theta, new_counter)

        return theta_trans

    return theta


global theta_record
theta_record =[]


def train_negative_log():
    train_neg_log = []

    for k in range(len(theta_record)):
        theta_k = theta_record[k]
        theta_kk = np.asarray(theta_k)
        summ = 0
        for i in range(len(x_matr_train)):
            a = (-y_train[i])*(np.dot(theta_kk.transpose(), x_matr_train[i]))
            b = np.log(1+np.exp(np.dot(theta_kk.transpose(), x_matr_train[i])))
            summ = summ + a + b
        train_neg_log.append(summ/len(x_matr_train))

    return train_neg_log


def valid_negative_log():
    valid_neg_log = []

    for k in range(len(theta_record)):
        theta_k = theta_record[k]
        theta_kk = np.asarray(theta_k)
        summ = 0
        for i in range(len(x_matr_valid)):
            a = (-y_valid[i])*(np.dot(theta_kk.transpose(), x_matr_valid[i]))
            b = np.log(1+np.exp(np.dot(theta_kk.transpose(), x_matr_valid[i])))
            summ = summ + a + b
        valid_neg_log.append(summ/len(x_matr_valid))

    return valid_neg_log


def loop_SGD(theta, counter, max_epoch):

    theta_record.append(theta)

    if counter < max_epoch:

        new_counter = counter + 1

        theta_trans2 = SGD(theta, 0)

        theta_trans_loop = loop_SGD(theta_trans2, new_counter, max_epoch)

        return theta_trans_loop

    return theta


final_theta = loop_SGD(theta, 0, int(num_epoch))
theta_record.pop(0)

prediction_train = []
for i in range(len(x_matr_train)):
    exp_xi = np.exp(np.dot(final_theta.transpose(), x_matr_train[i]))
    prob_i = (1 / (1 + exp_xi)) * (exp_xi)
    prob_i = round(prob_i)

    prediction_train.append(int(prob_i))

prediction_test = []
for i in range(len(x_matr_test)):
    exp_xi = np.exp(np.dot(final_theta.transpose(), x_matr_test[i]))
    prob_i = (1/(1+exp_xi))*(exp_xi)
    prob_i = round(prob_i)

    prediction_test.append(int(prob_i))

# writing train predictions
train_out = open(train_output, "w")
error_count_train = 0
for i_train in range(len(prediction_train)):
    train_out.write(str(prediction_train[i_train])+"\n")

    if str(prediction_train[i_train]) != str(y_train[i_train]):
        error_count_train += 1

# writing test predictions
test_out = open(test_output, "w")
error_count_test = 0
for i_test in range(len(prediction_test)):
    test_out.write(str(prediction_test[i_test]) + "\n")

    if str(prediction_test[i_test]) != str(y_test[i_test]):
        error_count_test += 1

train_errorR = error_count_train/len(prediction_train)
test_errorR = error_count_test/len(prediction_test)

# writing metrics file
metrics = open(metrics_out, "w")
metrics.write("error(train): "+str(train_errorR))
metrics.write("\n")
metrics.write("error(test): "+str(test_errorR))

train_log = train_negative_log()
train_log = train_log
valid_log = valid_negative_log()
xl = np.arange(0, int(num_epoch))
plt.plot(xl, train_log, label="Training data")
plt.plot(xl, valid_log, label="Validation data")
plt.xlabel("Number of epoch(s)")
plt.ylabel("Average negative log-likelihood")
plt.legend(loc='upper right')

plt.show()