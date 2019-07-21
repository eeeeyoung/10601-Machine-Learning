import numpy as np
import sys

# Training samples

arg_list = sys.argv[0:]

train_input = sys.argv[1]
test_input = sys.argv[2]
train_out = sys.argv[3]
test_out = sys.argv[4]
metrics_out = sys.argv[5]
ne = sys.argv[6]
hu = sys.argv[7]
infl = sys.argv[8]
learning_rate = sys.argv[9]

# train_data is a list assembling all training data with each entry storing
# a linearized vector of 128 float
train_data = []
real_label = []
test_data = []
test_label = []

# Opening and Reading from train and test data
test = open(test_input, "r")
train = open(train_input, "r")
for line in train:
    data = line.strip().split(",")
    real_label.append(int(data[0]))

    in_size = len(data[1:])
    vec = np.zeros(in_size)
    for i in range(in_size):
        vec[i] = int(data[i+1])
    train_data.append(vec)

for line in test:
    data_t = line.strip().split(",")
    test_label.append(int(data_t[0]))

    in_size_t = len(data_t[1:])
    vec = np.zeros(in_size_t)
    for i in range(in_size_t):
        vec[i] = int(data_t[i+1])
    test_data.append(vec)

hidden_units = int(hu)

init_flag = int(infl)

alpha1 = np.zeros((hidden_units, in_size+1))
for i in range(hidden_units):
    alpha1[i][0] = 0
beta1 = np.zeros((10, hidden_units+1))
for i in range(10):
    beta1[i][0] = 0

if init_flag == 1:
    alpha1_ = np.random.rand(hidden_units, in_size+1)
    alpha1 = (alpha1_/5)-0.1
    for i in range(hidden_units):
        alpha1[i][0] = 1

    beta1_ = np.random.rand(10, hidden_units+1)
    beta1 = (beta1_/5)-0.1
    for i in range(10):
        beta1[i][0] = 1


# In this design, sigmoid func takes a numpy array
def sigmoid(nparray):
    return 1/(1+np.exp(-nparray))


def J_func(y, y_hat):
    J_ = np.dot(y, np.log(y_hat))
    return J_


def NNForward_Backprop(index):
    y_ = real_label[index]
    y1 = np.zeros(10)
    y1[y_] = 1
    y = y1

    x = np.ones(in_size+1)
    x[1:len(x):] = train_data[index]

    # x is now the stochastically chosen input array with bias term x0 = 1
    inter = np.zeros(hidden_units)
    for i in range(hidden_units):
        inter[i] = np.dot(alpha1[i], x.transpose())

    z = np.ones(hidden_units+1)
    z_no_bias = sigmoid(inter)
    z[1:len(z):] = z_no_bias

    b = np.zeros(10)
    for i in range(10):
        b[i] = np.dot(beta1[i], z)

    inter2 = np.zeros(10)
    for i in range(10):
        inter2[i] = np.exp(b[i])

    y_hat = np.zeros(10)
    sum_inter2 = sum(inter2)

    for i in range(10):
        y_hat[i] = inter2[i]/sum_inter2

    gy = -y/y_hat

    gb = np.dot(gy.transpose(), (np.diag(y_hat)-np.outer(y_hat, y_hat.transpose())))
    gbeta = np.outer(gb, z.transpose())

    gz = np.dot(beta1.transpose(), gb)

    gz = gz[1:]
    z = z[1:]
    ga = gz*z*(1-z)

    galpha = np.outer(ga, x.transpose())
    return galpha, gbeta



# Making predictions on test data

# Prediction is reused after every epoch
def test_prediction(index):
    y_ = test_label[index]
    y1 = np.zeros(10)
    y1[y_] = 1
    y = y1
    global alpha1, beta1, J_test

    x = np.ones(in_size+1)
    x[1:len(x):] = test_data[index]

    # x is now the stochastically chosen input array with bias term x0 = 1
    inter = np.zeros(hidden_units)
    for i in range(hidden_units):
        inter[i] = np.dot(x.transpose(), alpha1[i])

    z = np.ones(hidden_units+1)
    z_no_bias = sigmoid(inter)
    z[1:len(z):] = z_no_bias

    b = np.zeros(10)
    for i in range(10):
        b[i] = np.dot(z.transpose(), beta1[i])

    inter2 = np.zeros(10)
    for i in range(10):
        inter2[i] = np.exp(b[i])

    y_hat = np.zeros(10)
    sum_inter2 = sum(inter2)

    for i in range(10):
        y_hat[i] = inter2[i]/sum_inter2

    summ = J_func(y, y_hat)

    return summ


def train_prediction(index):
    y_ = real_label[index]
    y1 = np.zeros(10)
    y1[y_] = 1
    y = y1
    global alpha1, beta1, J_train

    x = np.ones(in_size+1)
    x[1:len(x):] = train_data[index]

    # x is now the stochastically chosen input array with bias term x0 = 1
    inter = np.zeros(hidden_units)
    for i in range(hidden_units):
        inter[i] = np.dot(x.transpose(), alpha1[i])

    z = np.ones(hidden_units+1)
    z_no_bias = sigmoid(inter)
    z[1:len(z):] = z_no_bias

    b = np.zeros(10)
    for i in range(10):
        b[i] = np.dot(z.transpose(), beta1[i])

    inter2 = np.zeros(10)
    for i in range(10):
        inter2[i] = np.exp(b[i])

    y_hat = np.zeros(10)
    sum_inter2 = sum(inter2)

    for i in range(10):
        y_hat[i] = inter2[i]/sum_inter2

    summ = J_func(y, y_hat)

    return summ


def calculate_J():
    J_test = 0
    J_train = 0
    for i in range(len(test_data)):
        a = test_prediction(i)
        J_test += a

    for i in range(len(train_data)):
        a = train_prediction(i)
        J_train += a

    Avg_J_train = (-(1/len(train_data))*J_train)
    Avg_J_test = (-(1/len(test_data))*J_test)

    return Avg_J_train, Avg_J_test


mo = open(metrics_out, "w")


def mainSGD(num):
    global alpha1, beta1
    for i in range(num):
        for j in range(len(train_data)):
            gal, gbe = NNForward_Backprop(j)
            alpha1 = alpha1 - float(learning_rate)*gal
            beta1 = beta1 - float(learning_rate)*gbe

        avg_J_train, avg_J_test = calculate_J()
        mo.write("epoch="+str(i+1)+" "+"crossentropy(train): "+str(avg_J_train)+"\n")
        mo.write("epoch="+str(i+1)+" "+"crossentropy(test): "+str(avg_J_test)+"\n")


num_epoch = int(ne)
mainSGD(num_epoch)


trainerror_train = 0
testerror_count = 0


prediction_list = []
train_prediction_list = []


def final_test_prediction(index):
    y_ = test_label[index]
    y1 = np.zeros(10)
    y1[y_] = 1
    y = y1
    global alpha1, beta1, J_test

    x = np.ones(in_size+1)
    x[1:len(x):] = test_data[index]

    # x is now the stochastically chosen input array with bias term x0 = 1
    inter = np.zeros(hidden_units)
    for i in range(hidden_units):
        inter[i] = np.dot(x.transpose(), alpha1[i])

    z = np.ones(hidden_units+1)
    z_no_bias = sigmoid(inter)
    z[1:len(z):] = z_no_bias

    b = np.zeros(10)
    for i in range(10):
        b[i] = np.dot(z.transpose(), beta1[i])

    inter2 = np.zeros(10)
    for i in range(10):
        inter2[i] = np.exp(b[i])

    y_hat = np.zeros(10)
    sum_inter2 = sum(inter2)

    for i in range(10):
        y_hat[i] = inter2[i]/sum_inter2

    summ = J_func(y, y_hat)

    global prediction_list

    predict = np.argmax(y_hat)
    prediction_list.append(predict)


def final_train_prediction(index):
    y_ = real_label[index]
    y1 = np.zeros(10)
    y1[y_] = 1
    y = y1
    global alpha1, beta1, J_train

    x = np.ones(in_size+1)
    x[1:len(x):] = train_data[index]

    # x is now the stochastically chosen input array with bias term x0 = 1
    inter = np.zeros(hidden_units)
    for i in range(hidden_units):
        inter[i] = np.dot(x.transpose(), alpha1[i])

    z = np.ones(hidden_units+1)
    z_no_bias = sigmoid(inter)
    z[1:len(z):] = z_no_bias

    b = np.zeros(10)
    for i in range(10):
        b[i] = np.dot(z.transpose(), beta1[i])

    inter2 = np.zeros(10)
    for i in range(10):
        inter2[i] = np.exp(b[i])

    y_hat = np.zeros(10)
    sum_inter2 = sum(inter2)

    for i in range(10):
        y_hat[i] = inter2[i]/sum_inter2

    global train_prediction_list

    predict = np.argmax(y_hat)
    train_prediction_list.append(predict)


for i in range(len(test_data)):
    final_test_prediction(i)

for i in range(len(train_data)):
    final_train_prediction(i)

test_out = open(test_out, "w")
for k in range(len(prediction_list)):

    test_out.write(str(prediction_list[k])+"\n")

    if prediction_list[k] != test_label[k]:
        testerror_count += 1

train_out = open(train_out, "w")
for k in range(len(train_prediction_list)):

    train_out.write(str(train_prediction_list[k])+"\n")

    if train_prediction_list[k] != real_label[k]:
        trainerror_train += 1


testerror_rate = testerror_count/len(prediction_list)
trainerror_rate = trainerror_train/len(train_prediction_list)

mo.write("error(train): "+str(trainerror_rate)+"\n")
mo.write("error(test): "+str(testerror_rate)+"\n")