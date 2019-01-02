import sys
import math
import numpy as np

if __name__ == "__main__":

    arguments = sys.argv[1:]
    train_input = arguments[0]
    test_input = arguments[1]
    max_depth = int(arguments[2])
    train_output = arguments[3]
    test_output = arguments[4]
    metrics_output = arguments[5]


    data_with_label = np.genfromtxt(train_input, delimiter=',', unpack=True, dtype=None)
    #  size of each attribute
    #  assuming that the dataset is complete
    n = len(list(data_with_label[:, 1:][0]))

    #  number of attributes
    #  excluding the last column(i.e. results)
    m = len(data_with_label[0, :])-1


def entropy_cal(list_for_class):
    element_type = set(list_for_class)
    element_count = {}
    p_element = {}
    entropy = 0
    for ele in element_type:
        element_count[ele] = list(list_for_class).count(ele)
        p_element[ele] = element_count[ele]/len(list_for_class)
        entropy += -1 * p_element[ele] * math.log2(p_element[ele])
    return entropy


def joint_entropy_cal(list_for_class, list_for_attribute):
    #
    #  Treating classification column
    #
    element_type = set(list_for_class)
    element_count = {}
    p_element = {}
    entropy_class = 0

    for ele in element_type:
        element_count[ele] = list(list_for_class).count(ele)
        p_element[ele] = element_count[ele]/len(list_for_class)
        entropy_class += -1 * p_element[ele] * math.log2(p_element[ele])
    #
    # Initializing attribute_dictionary
    #
    attribute_type = set(list_for_attribute)
    attribute_count = {}
    p_attribute = {}
    conditional_prob_dict = {}
    conditional_entropy_dict = {}
    for ele1 in attribute_type:
        attribute_count[ele1] = list(list_for_attribute).count(ele1)
        p_attribute[ele1] = attribute_count[ele1]/len(list_for_attribute)
        #
        #   Addressing conditional_entropy
        #
        ent_sum = 0
        for ele in element_type:
            counter = 0
            for i in range(0, len(list(list_for_class))):
                if list(list_for_attribute)[i] == ele1 and list(list_for_class)[i] == ele:
                    counter += 1
            conditional_prob_dict[(ele, ele1)] = counter/attribute_count[ele1]
            prob = conditional_prob_dict[(ele, ele1)]
            if prob == 0:
                conditional_entropy_dict[(ele, ele1)] = 0
            else:
                conditional_entropy_dict[(ele, ele1)] = -prob*math.log2(prob)
            ent_sum += conditional_entropy_dict[(ele, ele1)]

        conditional_entropy_dict[ele1] = ent_sum
    #
    #   Addressing cumulative entropy given a specific attribute and the prob density of the attribute
    #
    joint_entropy = 0
    for ele1 in attribute_type:
        joint_entropy += conditional_entropy_dict[ele1]*p_attribute[ele1]

    return joint_entropy


def error_rate(dataset):
    data = dataset[:, 1:]
    class_array = data[-1]
    class_list = list(class_array)
    elements = set(class_list)
    element_count = {}
    for ele in elements:
        element_count[ele] = list(class_list).count(ele)
    major_element = max(element_count, key=element_count.get)
    error_rt = 1 - element_count[major_element]/len(class_list)
    return error_rt, major_element


#  for small_train.csv
#  number of attributes should be dynamic m = len(..)
#  size of each attribute set: n = 28

def tree_split(dataset, depth, current_node):
    if len(dataset[:, 0]) > 1:
        data = dataset[:, 1:]
        if depth >= max_depth:
            pass
        else:
            m = len(dataset)-1
            mutual_information_incol = {}
            for i in range(0, m):
                mutual_information_incol[i] = entropy_cal(data[-1]) - joint_entropy_cal(data[-1], data[i])
            max_mi_col = max(mutual_information_incol, key=mutual_information_incol.get)
            #
            #  splitting the dataset into 2, depending on attributes of max_mi_col
            #  only split if mi is > 0
            if mutual_information_incol[max_mi_col] == 0:
                pass
            else:
                dataset_0 = np.delete(dataset, max_mi_col, axis=0)
                dataset_1 = np.delete(dataset, max_mi_col, axis=0)
                delete_list1 = []
                delete_list0 = []

                attr_set = set(data[max_mi_col])
                attr_list = list(attr_set)
                attr0 = attr_list[0]
                attr1 = attr_list[1]
                for i in range(0, len(data[0])):
                    if data[max_mi_col][i] == attr_list[0]:
                        delete_list1.append(i+1)
                    elif data[max_mi_col][i] == attr_list[1]:
                        delete_list0.append(i+1)

                dataset_0 = np.delete(dataset_0, delete_list0, axis= 1)
                dataset_1 = np.delete(dataset_1, delete_list1, axis= 1)

                new_depth = depth + 1
        #
        # implement recursive function
        #
        #
        # define nodes of next level
        #
                label0 = dataset[max_mi_col][0]
                label1 = dataset[max_mi_col][0]
                er0, prediction0 = error_rate(dataset_0)
                er1, prediction1 = error_rate(dataset_1)
                current_node.lc = Node(dataset_0, new_depth, label0, er0, attr0, prediction0)
                current_node.rc = Node(dataset_1, new_depth, label1, er1, attr1, prediction1)

                tree_split(dataset_0, new_depth, current_node.lc)
                tree_split(dataset_1, new_depth, current_node.rc)


#
#  The main function first calls a majority vote on the whole dataset
#  then it calls for calculation to determine best mutual information, then split
#
class Node:
    def __init__(self, datalist, current_depth, label, er, attr, prediciton):
        self.depth = current_depth
        self.dataset = datalist
        self.er = er
        self.label = label
        self.attr = attr
        self.prediction = prediciton

        self.lc = None
        self.rc = None


def main(dataset):
    current_depth = 0
    tree_split(dataset, current_depth, root)


def DFS_for_tree_output(node):
    if node == root:
        class_data = node.dataset[-1, 1:]
        counter0 = 0
        counter1 = 0
        for element in class_data:
            if element == classes[0]:
                counter0 += 1
            if element == classes[1]:
                counter1 += 1
        print("[", counter0, classes[0].decode('utf-8'), "/", counter1, classes[1].decode('utf-8'), "]")
    else:
        class_label = node.dataset[:, 0][-1]
        class_data = node.dataset[-1, 1:]
        counter0 = 0
        counter1 = 0
        for element in class_data:
            if element == classes[0]:
                counter0 += 1
            if element == classes[1]:
                counter1 += 1
        print("|"*node.depth, node.label.decode('utf-8'), " = ", node.attr.decode('utf-8'), ":", "[", counter0, classes[0].decode('utf-8'), "/", counter1, classes[1].decode('utf-8'), "]")

    if node.lc != None:
        DFS_for_tree_output(node.lc)
        if node.rc != None:
            DFS_for_tree_output(node.rc)


def loop_thru(data, output_file, trainORtest):
    size = len(list(data[:, 1:][0]))
    prediction_output_list = []
    for i in range(size):
        prediction_row = data[:, 1:][:, i]

        #  calling function prediction()

        prediction_output = prediction(prediction_row, root)
        prediction_output_list.append(prediction_output)
    f = open(output_file, 'w')
    for ele in prediction_output_list:
        ele = ele.decode('utf-8')
        f.write(ele)
        f.write("\n")
    f.close()
    if trainORtest == "train":
        error_count = 0
        m0 = open(metrics_output, 'w')
        for i in range(n):
            if prediction_output_list[i] != data[:, 1:][-1][i]:
                error_count += 1
            else:
                pass
        er = error_count/n
        m0.write("error(train): "+str(er))
        m0.write("\n")
    elif trainORtest == "test":
        error_count = 0
        m0 = open(metrics_output, 'a')
        for i in range(len(data[:, 1:][-1])):
            if prediction_output_list[i] != data[:, 1:][-1][i]:
                error_count += 1
            else:
                pass
        er = error_count/len(prediction_output_list)
        m0.write("error(test): "+str(er))
        m0.write("\n")


all_labels = list(data_with_label[:, 0])


def prediction(data, node):
    if node.lc is not None:
        label = node.lc.label
        for i in range(len(all_labels)):
            if all_labels[i] == label:
                check_label = i
        attribute = data[check_label]
        if str(attribute) == str(node.lc.attr):
            answer = prediction(data, node.lc)
        elif str(attribute) == str(node.rc.attr):
            answer = prediction(data, node.rc)
    elif node.lc is None:
        return node.prediction
    return answer


class_data = data_with_label[-1, 1:]
classes = list(set(class_data))
root_er, prediction_root = error_rate(data_with_label)
root = Node(data_with_label, 0, "root", root_er, 0, prediction_root)

# running the programme
main(data_with_label)
DFS_for_tree_output(root)  # this prints out the tree


# prediction_on_train: output to trainlabel
loop_thru(data_with_label, train_output, "train")

# prediction on test: output to testlabel
test_data = np.genfromtxt(test_input, delimiter=',', unpack=True, dtype=None)
loop_thru(test_data, test_output, "test")

# metrics about the test prediciton accuracy
#  Now included in loop_thru