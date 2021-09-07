import os
import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
from multiprocessing import Pool

from func_module import freqoracle
from func_module import errormetric
from func_module import realfreq

# calculate theta
def theta_calculation(ahead_tree_height, epsilon, user_scale, branch, data_dimension):
    domain_combination_number = data_dimension * (data_dimension-1) / 2
    user_scale_in_each_layer = user_scale / (ahead_tree_height*domain_combination_number)
    varience_of_oue = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((math.prod(branch) + 1) * varience_of_oue)

# construct sub-domain partition vectors
def construct_translation_vector(domain_size, branch):
    translation_vector = []
    for i in range(branch[0]):
        for j in range(branch[1]):
                translation_vector.append(np.array(
                    [i * domain_size[0] // branch[0], i * domain_size[0] // branch[0],
                     j * domain_size[1] // branch[1], j * domain_size[1] // branch[1]]))
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

# Step1: User Partition (UP) in Section 4.2
def user_record_partition(dataset, ahead_tree_height, domain_size, data_dimension):
    domain_combination_number = int(data_dimension*(data_dimension-1)/2)
    domain_index_list = []
    for i in range(data_dimension - 1):
        for j in range(i + 1, data_dimension):
            domain_index_list.append([i, j])

    user_histogram = np.zeros((domain_combination_number, ahead_tree_height, domain_size[0], domain_size[1]), dtype=np.int32)

    for k, item in enumerate(dataset):
        user_sample_tree_id = np.random.randint(0, domain_combination_number)
        user_sample_layer_id = np.random.randint(0, ahead_tree_height)
        user_histogram[user_sample_tree_id, user_sample_layer_id, item[domain_index_list[user_sample_tree_id][0]], item[domain_index_list[user_sample_tree_id][1]]] += 1
    return user_histogram

# Step2: New Decomposition Generation (NDG) in Section 4.2
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
                node_frequency = 0
                node_divide_flag = True
                node_count = 0
                node_interval = TempItem0 + item1
                node_name = str(node_interval)
                ahead_tree.create_node(node_name, node_name, parent=node.identifier, data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
                layer_index += 1

# Step3: Noisy Frequency Construction (NFC) in Section 4.2
def ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch, translation_vector, user_dataset_partition, epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # update ahead_tree structrue 
        ahead_tree_update(ahead_tree, tree_height, theta, branch, translation_vector, layer_index)
        # update sub-domain partition vectors
        translation_vector[:] = translation_vector[:] // np.array([branch[0], branch[0], branch[1], branch[1]])
        translation_vector = duplicate_remove(translation_vector)
        # update ahead_tree sub-domain frequency 
        node_frequency_aggregation(ahead_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1

# Step4: Post-processing (PP) in Section 4.2
def ahead_tree_postprocessing(ahead_tree):
    lowest_nodes_number = 0
    for _, node in reversed(list(enumerate(ahead_tree.all_nodes()))):
        if lowest_nodes_number < ahead_tree.size(ahead_tree.depth()):
            lowest_nodes_number += 1
            continue

        if ahead_tree.depth(node) != ahead_tree.depth() and ahead_tree.children(node.identifier) != []:
            numerator = 0
            children_frequency_sum = 0
            for j, child_node in enumerate(ahead_tree.children(node.identifier)):
                numerator += 1 / child_node.data.count
                children_frequency_sum += child_node.data.frequency

            denominator = numerator + 1
            coeff0 = numerator / denominator
            coeff1 = 1 - coeff0

            node.data.frequency = coeff0 * node.data.frequency + coeff1 * children_frequency_sum
            node.data.count = 1/coeff0

# Step 2: Consistency on Attributes in Appendix F
def ahead_tree_consistency_step(ahead_forest, ahead_tree_height, domain_size, branch, domain_combination_list):
    interval_length = domain_size/branch
    ahead_forest_frequency_distribution_list = []
    for i in range(1, ahead_tree_height+1):
        layer_frequency_distribution_list = []
        layer_frequency_count_list = []
        for j, ahead_tree in enumerate(ahead_forest):
            layer_frequency_distribution = np.zeros(branch ** i)
            layer_frequency_count = np.zeros(branch ** i)

            for k, node in enumerate(ahead_tree.all_nodes()):
                if ahead_tree.level(node.identifier) == i:
                    d1_index = int(node.data.interval[0] // interval_length[0])
                    d2_index = int(node.data.interval[2] // interval_length[1])
                    layer_frequency_distribution[d1_index, d2_index] = node.data.frequency
                    layer_frequency_count[d1_index, d2_index] = node.data.count

            layer_frequency_distribution_list.append(layer_frequency_distribution)
            layer_frequency_count_list.append(layer_frequency_count)


        # operate attribute by attribute
        for m in range(len(domain_size)):
            layer_frequency_distribution_index_list = []

            # attribute location
            for n, domain_combination in enumerate(domain_combination_list):
                if m in domain_combination:
                    layer_frequency_distribution_index_list.append(n)

            # frequency summation
            for s in range(branch[0] ** i):
                temp_frequency_count = 0
                temp_frequency_distribution = 0
                for index in layer_frequency_distribution_index_list:
                    if m == domain_combination_list[index][0]:
                        temp_frequency_count = temp_frequency_count + layer_frequency_count_list[index][s, :].sum()
                    elif m == domain_combination_list[index][1]:
                        temp_frequency_count = temp_frequency_count + layer_frequency_count_list[index][:, s].sum()

                for index in layer_frequency_distribution_index_list:
                    if m == domain_combination_list[index][0]:
                        temp_frequency_distribution = temp_frequency_distribution + (layer_frequency_count_list[index][s, :].sum() / temp_frequency_count) * layer_frequency_distribution_list[index][s, :].sum()
                    if m == domain_combination_list[index][1]:
                        temp_frequency_distribution = temp_frequency_distribution + (layer_frequency_count_list[index][:, s].sum() / temp_frequency_count) * layer_frequency_distribution_list[index][:, s].sum()

                for index in layer_frequency_distribution_index_list:
                    if m == domain_combination_list[index][0]:
                        diff_frequency_distribution = temp_frequency_distribution - layer_frequency_count_list[index][s, :].sum()
                        layer_frequency_count_list[index][s, :] = layer_frequency_count_list[index][s, :] + diff_frequency_distribution/branch[0]**i
                    elif m == domain_combination_list[index][1]:
                        diff_frequency_distribution = temp_frequency_distribution - layer_frequency_count_list[index][:, s].sum()
                        layer_frequency_count_list[index][s, :] = layer_frequency_count_list[index][:, s] + diff_frequency_distribution / branch[0] ** i

        ahead_forest_frequency_distribution_list.append(layer_frequency_distribution_list)
        interval_length = interval_length / branch
    ahead_forest_frequency_distribution_list_transfer_to_ahead_forest = []
    
    for v in range(len(domain_combination_list)):
        ahead_tree_list = []
        for g in range(ahead_tree_height):
            ahead_tree_list.append(ahead_forest_frequency_distribution_list[g][v])

        ahead_forest_frequency_distribution_list_transfer_to_ahead_forest.append(ahead_tree_list)
        
    return ahead_forest_frequency_distribution_list_transfer_to_ahead_forest

# Step 3: Maximum Entropy Optimization in Appendix F and record query errors
def ahead_tree_query_error_recorder(ahead_forest, ahead_tree_list, real_frequency, query_interval_table, domain_size, MSEDict, domain_combination_list, data_dimension):
    errList = np.zeros(len(query_interval_table))
    for i, query_interval in enumerate(query_interval_table):
        real_frequency_value = real_frequency[i]
        vector_length = 2 ** data_dimension
        two_dim_frequency_value_list = []
        estimated_frequency_value_vector_index_list = []

        for j, domain_combination in enumerate(domain_combination_list):
            domain_combination_query = np.array([domain_combination[0]*2, domain_combination[0]*2 + 1, domain_combination[1]*2, domain_combination[1]*2 + 1])
            two_dim_frequency_value_list.append(ahead_tree_answer_query00(ahead_forest[j], ahead_tree_list[j], query_interval[domain_combination_query], domain_size))
            estimated_frequency_value_vector_index = []
            for vector_index in range(vector_length):
                if not (vector_index >> int(data_dimension - 1 - domain_combination[0]) & 1) and not (
                        vector_index >> int(data_dimension - 1 - domain_combination[1]) & 1):
                    estimated_frequency_value_vector_index.append(vector_index)
            estimated_frequency_value_vector_index_list.append(estimated_frequency_value_vector_index)

            two_dim_frequency_value_list.append(ahead_tree_answer_query01(ahead_forest[j], ahead_tree_list[j], query_interval[domain_combination_query], domain_size))
            estimated_frequency_value_vector_index = []
            for vector_index in range(vector_length):
                if not (vector_index >> int(data_dimension - 1 - domain_combination[0]) & 1) and (
                        vector_index >> int(data_dimension - 1 - domain_combination[1]) & 1):
                    estimated_frequency_value_vector_index.append(vector_index)
            estimated_frequency_value_vector_index_list.append(estimated_frequency_value_vector_index)

            two_dim_frequency_value_list.append(ahead_tree_answer_query10(ahead_forest[j], ahead_tree_list[j], query_interval[domain_combination_query], domain_size))
            estimated_frequency_value_vector_index = []
            for vector_index in range(vector_length):
                if (vector_index >> int(data_dimension - 1 - domain_combination[0]) & 1) and not (
                        vector_index >> int(data_dimension - 1 - domain_combination[1]) & 1):
                    estimated_frequency_value_vector_index.append(vector_index)
            estimated_frequency_value_vector_index_list.append(estimated_frequency_value_vector_index)

            two_dim_frequency_value_list.append(ahead_tree_answer_query11(ahead_forest[j], ahead_tree_list[j], query_interval[domain_combination_query], domain_size))
            estimated_frequency_value_vector_index = []
            for vector_index in range(vector_length):
                if (vector_index >> int(data_dimension - 1 - domain_combination[0]) & 1) and (
                        vector_index >> int(data_dimension - 1 - domain_combination[1]) & 1):
                    estimated_frequency_value_vector_index.append(vector_index)
            estimated_frequency_value_vector_index_list.append(estimated_frequency_value_vector_index)

        estimated_frequency_value_vector = np.ones(vector_length) * 1/vector_length
        temp_value_vector = np.ones(vector_length) * 1/vector_length

        for t in range(len(two_dim_frequency_value_list)):
            if two_dim_frequency_value_list[t] < 0:
                two_dim_frequency_value_list[t] = 0

        diff_estimated_frequency_value = np.ones(vector_length)
        while abs(diff_estimated_frequency_value).sum() > 1e-5:
            for p, frequency_value in enumerate(two_dim_frequency_value_list):
                frequency_value_sum = estimated_frequency_value_vector[estimated_frequency_value_vector_index_list[p]].sum()
                if frequency_value_sum == 0:
                    frequency_value_sum = 1e-5
                estimated_frequency_value_vector[estimated_frequency_value_vector_index_list[p]] = (estimated_frequency_value_vector[estimated_frequency_value_vector_index_list[p]]/frequency_value_sum) * frequency_value
            diff_estimated_frequency_value = estimated_frequency_value_vector - temp_value_vector
            temp_value_vector = estimated_frequency_value_vector.copy()
        errList[i] = real_frequency_value - estimated_frequency_value_vector[0]
        print('answer index {}-th query'.format(i))
        print("real_frequency_value: ", real_frequency_value)
        print("estimated_frequency_value: ", estimated_frequency_value_vector[0])

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def add_children_nodes(node, ahead_tree, branch, ahead_tree_height):
    tree_fanout = math.prod(branch)
    if ahead_tree.depth(node) == ahead_tree_height:
        pass
    else:
        for i in range(branch[0]):
            for j in range(branch[1]):

                node_frequency = node.data.frequency/tree_fanout
                node_divide_flag = False
                node_count = node.data.count * tree_fanout
                d1_left = node.data.interval[0]
                d1_right = node.data.interval[1]
                d2_left = node.data.interval[2]

                interval_length = (d1_right - d1_left)/branch[0]

                node_interval = [d1_left + interval_length * i, d1_left + interval_length * (i + 1), d2_left + interval_length * j, d2_left + interval_length * (j + 1)]
                node_name = str(node_interval)
                ahead_tree.create_node(node_name, node_name, parent=node.identifier, data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
                add_children_nodes(ahead_tree.nodes[node_name], ahead_tree, branch, ahead_tree_height)


def restore_the_complete_tree_structure(ahead_tree, branch, ahead_tree_height):
    for k, node in enumerate(ahead_tree.all_nodes()):
        if ahead_tree.children(node.identifier) != []:
            continue
        if ahead_tree.depth(node) != ahead_tree_height and ahead_tree.children(node.identifier) == []:
            add_children_nodes(node, ahead_tree, branch, ahead_tree_height)


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
        user_record_list.append(user_dataset[d1_left:d1_right, d2_left:d2_right].sum())

    noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32), sum(user_record_list))
    noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p, q)

    for i, node in enumerate(ahead_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (node.data.count + 1)
            node.data.count += 1


def ahead_tree_answer_query00(ahead_tree, ahead_tree_list, query_interval, domain_size):
    estimated_frequency_value = 0
    # set 2-dim range query
    query_interval_temp = np.zeros(domain_size)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])


    query_interval_temp[query_d1_left:query_d1_right, query_d2_left:query_d2_right] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        depth = ahead_tree.depth(node.identifier)-1
        query_interval_length = d2_right - d2_left

        query_interval_temp_sum = query_interval_temp[d1_left:d1_right, d2_left:d2_right].sum()
        query_interval_area = (d2_right-d2_left)*(d1_right-d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + ahead_tree_list[depth][int(d1_left/query_interval_length), int(d2_left/query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0
            continue

        if not ahead_tree.children(node.identifier):
            coeff = query_interval_temp_sum/query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * ahead_tree_list[depth][int(d1_left/query_interval_length), int(d2_left/query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0

    return estimated_frequency_value


def ahead_tree_answer_query01(ahead_tree, ahead_tree_list, query_interval, domain_size):
    estimated_frequency_value = 0
    # set 2-dim range query
    query_interval_temp = np.zeros(domain_size)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])

    query_interval_temp[query_d1_left:query_d1_right, 0:query_d2_left] = 1
    query_interval_temp[query_d1_left:query_d1_right, query_d2_right:] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        depth = ahead_tree.depth(node.identifier) - 1
        query_interval_length = d2_right - d2_left

        query_interval_temp_sum = query_interval_temp[d1_left:d1_right, d2_left:d2_right].sum()
        query_interval_area = (d2_right-d2_left)*(d1_right-d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0
            continue

        if not ahead_tree.children(node.identifier):
            coeff = query_interval_temp_sum/query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0

    return estimated_frequency_value


def ahead_tree_answer_query10(ahead_tree, ahead_tree_list, query_interval, domain_size):
    estimated_frequency_value = 0
    # set 2-dim range query
    query_interval_temp = np.zeros(domain_size)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])
    query_interval_temp[0:query_d1_left, query_d2_left:query_d2_right] = 1
    query_interval_temp[query_d1_right:, query_d2_left:query_d2_right] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        depth = ahead_tree.depth(node.identifier) - 1
        query_interval_length = d2_right - d2_left

        query_interval_temp_sum = query_interval_temp[d1_left:d1_right, d2_left:d2_right].sum()
        query_interval_area = (d2_right-d2_left)*(d1_right-d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0

        if not ahead_tree.children(node.identifier):
            coeff = query_interval_temp_sum / query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0


    return estimated_frequency_value


def ahead_tree_answer_query11(ahead_tree, ahead_tree_list, query_interval, domain_size):
    estimated_frequency_value = 0
    # set 2-dim range query
    query_interval_temp = np.zeros(domain_size)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])

    query_interval_temp[0:query_d1_left, 0:query_d2_left] = 1
    query_interval_temp[0:query_d1_left, query_d2_right:] = 1
    query_interval_temp[query_d1_right:, 0:query_d2_left] = 1
    query_interval_temp[query_d1_right:, query_d2_right:] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        depth = ahead_tree.depth(node.identifier) - 1
        query_interval_length = d2_right - d2_left

        query_interval_temp_sum = query_interval_temp[d1_left:d1_right, d2_left:d2_right].sum()
        query_interval_area = (d2_right-d2_left)*(d1_right-d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0

        if not ahead_tree.children(node.identifier):
            coeff = query_interval_temp_sum / query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * ahead_tree_list[depth][
                int(d1_left / query_interval_length), int(d2_left / query_interval_length)]
            query_interval_temp[d1_left:d1_right, d2_left:d2_right] = 0

    return estimated_frequency_value


def main_fun(repeat_time, domain_size, branch, ahead_tree_height, theta, real_frequency, query_interval_table, epsilon, data_name, data_size_name, domain_name, data_dimension, dataset, data_size):
    repeat = 0
    MSEDict = {'rand': []}

    while repeat < repeat_time:
        ahead_forest = []
        # user partition
        user_dataset_partition = user_record_partition(dataset, ahead_tree_height, domain_size, data_dimension)

        tree_id = 0
        domain_combination_list = []

        # build a 2D ahead_tree forest
        for i in range(data_dimension-1):
            for j in range(i+1, data_dimension):
                # initialize the tree structure, set the root node
                domain_combination_list.append([i, j])
                ahead_tree = Tree()
                ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size[0], 0, domain_size[1]])))

                # construct sub-domain partition vectors
                translation_vector = construct_translation_vector(domain_size, branch)
                
                # build a tree structure
                ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch, translation_vector, user_dataset_partition[tree_id], epsilon)

                # ahead_tree post-processing
                ahead_tree_postprocessing(ahead_tree)

                ahead_forest.append(ahead_tree)
                tree_id += 1

        # restore the complete 2-dim tree structure
        for k, ahead_tree in enumerate(ahead_forest):
            restore_the_complete_tree_structure(ahead_tree, branch, ahead_tree_height)

        # consistency on attributes
        ahead_tree_list = ahead_tree_consistency_step(ahead_forest, ahead_tree_height, domain_size, branch, domain_combination_list)

        # ahead_tree answer query
        ahead_tree_query_error_recorder(ahead_forest, ahead_tree_list, real_frequency, query_interval_table, domain_size, MSEDict, domain_combination_list, data_dimension)


        # record error
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv('rand_result/MSE_lle_ahead_branch{}-{}-{}-{}-{}-{}.csv'.format(branch[0],
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
    data_dimension = 5
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
    data_name = '5dim_laplace08' 
    data_size_name = 'set_10_6'
    domain_name = 'domain6_attribute{}'.format(data_dimension)
    
    # load dataset
    data_path = './dataset/{}-{}-{}-data.txt'.format(data_name, data_size_name, domain_name)
    dataset = np.loadtxt(data_path, np.int32)
    dataset = dataset[:, :data_dimension]
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
    theta = theta_calculation(ahead_tree_height, epsilon, data_size, branch[:2], data_dimension)
    
    # running 
    main_fun(repeat_time, domain_size[:2], branch[:2], ahead_tree_height, theta, real_frequency, query_interval_table, epsilon, data_name, data_size_name, domain_name, data_dimension, dataset, data_size)
    