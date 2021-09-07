import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
from multiprocessing import Pool
import time
import os

from func_module import freqoracle
from func_module import errormetric
from func_module import realfreq

# calculate theta
def theta_calculation(ahead_tree_height, epsilon, user_scale, branch):
    user_scale_in_each_layer = user_scale / ahead_tree_height
    varience_of_oue = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((math.prod(branch) + 1) * varience_of_oue)

# construct sub-domain partition vectors
def construct_translation_vector(domain_size, branch):
    translation_vector = []
    for i in range(branch[0]):
        for j in range(branch[1]):
            for k in range(branch[2]):
                translation_vector.append(np.array(
                    [i * domain_size[0] // branch[0], i * domain_size[0] // branch[0],
                     j * domain_size[1] // branch[1], j * domain_size[1] // branch[1],
                     k * domain_size[2] // branch[2], k * domain_size[2] // branch[2]]))
    return translation_vector

# remove duplicated sub-domain partition vectors
def duplicate_remove(list1):
    list2 = []
    for li1 in list1:
        Flag1 = True
        for li2 in list2:
            if (li1 == li2).all():
                Flag1 = False
                break
        if Flag1 == True:
            list2.append(li1)

    return list2

# Step1: User Partition (UP)
def user_record_partition(data_path, ahead_tree_height, domain_size):
    dataset = np.loadtxt(data_path, np.int32)
    user_sample_id = np.random.randint(0, ahead_tree_height, len(dataset)).reshape(len(dataset), 1)  # user sample list
    user_histogram = np.zeros((ahead_tree_height, domain_size[0], domain_size[1], domain_size[2]), dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[user_sample_id[k], item[0], item[1], item[2]] += 1
    return user_histogram

# Step2: New Decomposition Generation (NDG)
def ahead_tree_update(ahead_tree, tree_height, theta, branch, translation_vector, layer_index):
    for node in ahead_tree.leaves():
        if not node.data.divide_flag:
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (node.data.frequency < theta):
            node.data.divide_flag = False
            continue

        else:
            TempItem0 = np.zeros(node.data.interval.shape)
            for j in range(0, len(node.data.interval), 2):
                if node.data.interval[j + 1] - node.data.interval[j] > 1:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch[j//2] + node.data.interval[j]
                else:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = node.data.interval[j + 1]

            for item1 in translation_vector:
                node_name = str(tree_height) + str(layer_index)
                node_frequency = 0
                node_divide_flag = True
                node_count = 0
                node_interval = TempItem0 + item1
                ahead_tree.create_node(node_name, node_name, parent=node.identifier, data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
                layer_index += 1

# Step3: Noisy Frequency Construction (NFC)
def ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch, translation_vector, user_dataset_partition, epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # update ahead_tree structrue 
        ahead_tree_update(ahead_tree, tree_height, theta, branch, translation_vector, layer_index)
        # update sub-domain partition vectors
        translation_vector[:] = translation_vector[:] // np.array([branch[0], branch[0], branch[1], branch[1], branch[2], branch[2]])
        translation_vector = duplicate_remove(translation_vector)
        # update ahead_tree sub-domain frequency
        node_frequency_aggregation(ahead_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1

# Step4: Post-processing (PP)
def ahead_tree_postprocessing(ahead_tree):
    lowest_nodes_number = 0
    for _, node in reversed(list(enumerate(ahead_tree.all_nodes()))):
        if lowest_nodes_number < ahead_tree.size(ahead_tree.depth()):
            lowest_nodes_number += 1
            continue

        if ahead_tree.depth(node) != ahead_tree.depth() and ahead_tree.children(node.identifier) != []:
            numerator = 1 / node.data.count
            children_frequency_sum = 0
            for j, child_node in enumerate(ahead_tree.children(node.identifier)):
                numerator += 1 / child_node.data.count
                children_frequency_sum += child_node.data.frequency

            denominator = numerator + 1
            coeff0 = numerator / denominator
            coeff1 = 1 - coeff0

            node.data.frequency = coeff0 * node.data.frequency + coeff1 * children_frequency_sum
            node.data.count = 1/coeff0

# answer range queries
def ahead_tree_answer_query(ahead_tree, query_interval, domain_size):
    estimated_frequency_value = 0
    # set 3-dim range query
    query_interval_temp = np.zeros(domain_size, np.int8)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])
    query_d3_left = int(query_interval[4])
    query_d3_right = int(query_interval[5])
    query_interval_temp[query_d1_left:query_d1_right, query_d2_left:query_d2_right, query_d3_left:query_d3_right] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        d3_left = int(node.data.interval[4])
        d3_right = int(node.data.interval[5])

        query_interval_temp_sum = query_interval_temp[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right].sum()
        query_interval_area = (d3_right-d3_left)*(d2_right-d2_left)*(d1_right-d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + node.data.frequency
            query_interval_temp[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right] = 0
            continue

        if ahead_tree.children(node.identifier) == []:
            coeff = query_interval_temp_sum/query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * node.data.frequency
            query_interval_temp[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right] = 0

    return estimated_frequency_value

# record query errors
def ahead_tree_query_error_recorder(ahead_tree, real_frequency, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    for i, query_interval in enumerate(query_interval_table):
        real_frequency_value = real_frequency[i]
        estimated_frequency_value = ahead_tree_answer_query(ahead_tree, query_interval, domain_size)
        errList[i] = real_frequency_value - estimated_frequency_value
        print('answer index {}-th query'.format(i))
        print("real_frequency_value: ", real_frequency_value)
        print("estimated_frequency_value: ", estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def node_frequency_aggregation(ahead_tree, user_dataset, epsilon):
    # estimate the frequency values, and update the frequency values of the nodes
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))

    user_record_list = []
    for node in ahead_tree.leaves():
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        d3_left = int(node.data.interval[4])
        d3_right = int(node.data.interval[5])
        user_record_list.append(user_dataset[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right].sum())

    noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32), sum(user_record_list))
    noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p, q)

    for i, node in enumerate(ahead_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (node.data.count + 1)
            node.data.count += 1


def main_fun(repeat_time, domain_size, branch, ahead_tree_height, theta, real_frequency, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name):
    repeat = 0
    MSEDict = {'rand': []}
    while repeat < repeat_time:
        
        # user partition
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height, domain_size)
        
        # initialize the tree structure, set the root node
        ahead_tree = Tree()
        ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size[0], 0, domain_size[1], 0, domain_size[2]])))

        # construct sub-domain partition vectors
        translation_vector = construct_translation_vector(domain_size, branch)
        
        # build a tree structure
        ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch, translation_vector, user_dataset_partition, epsilon)

        # ahead_tree post-processing
        ahead_tree_postprocessing(ahead_tree)

        for i in range(ahead_tree.depth()+2):
            print(i, ahead_tree.size(level=i))
            
        # ahead_tree answer query
        ahead_tree_query_error_recorder(ahead_tree, real_frequency, query_interval_table, domain_size, MSEDict)

        # record errors
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv('rand_result/MSE_de_ahead_branch{}-{}-{}-{}-{}-{}.csv'.format(branch[0],
                                                                                          data_name,
                                                                                          data_size_name,
                                                                                          domain_name,
                                                                                          epsilon,
                                                                                          repeat_time))
        repeat += 1
        print("repeat time: ", repeat)


class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval


if __name__ == "__main__":    
    
    # set the number of repeated experiments
    repeat_time = 1
    
    # set privacy budget
    epsilon = 1
    
    # set data_dimension, branch and domain_size
    data_dimension = 3
    branch = 2
    domain_size = 2 ** 6
    ahead_tree_height = int(math.log(domain_size, branch))
    branch = np.ones(data_dimension, dtype=int) * branch
    domain_size = list(np.ones(data_dimension, dtype=int) * domain_size)
    
    # load query table
    query_path = './query_table/rand_query_domain6_attribute{}.txt'.format(data_dimension)
    query_interval_table = np.loadtxt(query_path, int)
    print("the top 5 range queries in query_interval_table: \n", query_interval_table[:5])
    
    # select dataset
    data_name = '3dim_laplace09'
    data_size_name = 'set_10_6'
    domain_name = 'domain6_attribute{}'.format(data_dimension)
    
    # load dataset
    data_path = './dataset/{}-{}-{}-data.txt'.format(data_name, data_size_name, domain_name)
    dataset = np.loadtxt(data_path, np.int32)
    print("the shape of dataset: ", dataset.shape)
    data_size = dataset.shape[0]
    
    # calculate/load true frequency
    real_frequency_path = './query_table/real_frequency-{}-{}-{}.npy'.format(data_name, data_size_name, domain_name)
    if os.path.exists(real_frequency_path):
        real_frequency = np.load(real_frequency_path)
    else:
        real_frequency = realfreq.real_frequency_generation(dataset, data_size, domain_size, data_dimension, query_interval_table)
        np.save(real_frequency_path, real_frequency)
        
    # calculate theta
    theta = theta_calculation(ahead_tree_height, epsilon, data_size, branch)
    print(theta)
    
    # running 
    main_fun(repeat_time, domain_size, branch, ahead_tree_height, theta, real_frequency, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name)
