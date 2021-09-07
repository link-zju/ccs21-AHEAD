import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
from multiprocessing import Pool
import time

from func_module import freqoracle
from func_module import errormetric

def duplicate_removal(lis1):
    # print('lis')
    # print(lis1)
    lis2 = []
    for li1 in lis1:
        Flag1 = True

        for li2 in lis2:
            if (li1 == li2).all():
                Flag1 = False
                break

        if Flag1 == True:
            lis2.append(li1)

    return lis2


def construct_translation_vector(domain_size, branch):
    # 构造平移向量
    translation_vector = []
    for i in range(branch[0]):
        for j in range(branch[1]):
                translation_vector.append(np.array(
                    [i * domain_size[0] // branch[0], i * domain_size[0] // branch[0],
                     j * domain_size[1] // branch[1], j * domain_size[1] // branch[1]]))
    return translation_vector


def ahead_tree_update(ahead_tree, tree_height, SNR, branch, translation_vector, layer_index):
    for node in ahead_tree.leaves():
        if not node.data.divide_flag:
            # node.data.count += 1
            # print("not node.data.divide_flag")
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (node.data.frequency < SNR):
            node.data.divide_flag = False
            # node.data.count += 1
            # print('node.data.frequency < SNR')
            continue

        else:
            TempItem0 = np.zeros(node.data.interval.shape)
            for j in range(0, len(node.data.interval), 2):
                # print(j)
                # print(TempItem0)
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

    # 这里写一个translationVector去重操作


def user_record(data_path, domain_size):
    dataset = np.loadtxt(data_path, np.int32)
    # print(dataset)
    # dataset = np.array(dataset['dataset'])
    user_histogram = np.zeros(domain_size, dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[item[0], item[1], item[2], item[3], item[4]] += 1
    return user_histogram


def user_record_partition(data_path, ahead_tree_height, domain_size, data_dimension):
    dataset = np.loadtxt(data_path, np.int32)
    # dataset = np.array(dataset['dataset'])
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


def ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition, epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # ahead_tree结构更新
        ahead_tree_update(ahead_tree, tree_height, SNR, branch, translation_vector, layer_index)
        # 更新平移向量
        translation_vector[:] = translation_vector[:] // np.array([branch[0], branch[0], branch[1], branch[1]])
        translation_vector = duplicate_removal(translation_vector)
        # ahead_tree频率更新
        node_frequency_aggregation(ahead_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1


def node_frequency_aggregation(ahead_tree, user_dataset, epsilon):

    # 统计频率，更新叶子结点的频率值
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


def ahead_tree_postprocessing(ahead_tree):
    lowest_nodes_number = 0
    for _, node in reversed(list(enumerate(ahead_tree.all_nodes()))):
        if lowest_nodes_number < ahead_tree.size(ahead_tree.depth()):
            # print(lowest_nodes_number)
            lowest_nodes_number += 1
            continue

        if ahead_tree.depth(node) != ahead_tree.depth() and ahead_tree.children(node.identifier) != []:
            # numerator = 1 / node.data.count
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


def ahead_tree_answer_query00(ahead_tree, ahead_tree_list, query_interval, domain_size):
    estimated_frequency_value = 0
    # 设置查询区域
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
    # 设置查询区域
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

        # 这个空间判断需要改写，不同情况不能简单的进行判断。
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
    # 设置查询区域
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

        # 这个空间判断需要改写，不同情况不能简单的进行判断。
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
    # 设置查询区域
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


def ahead_tree_query_error_recorder(ahead_forest, ahead_tree_list, user_dataset, query_interval_table, domain_size, MSEDict, domain_combination_list, data_dimension, branch):
    errList = np.zeros(len(query_interval_table))
    user_dataset_size = user_dataset.sum()

    for i, query_interval in enumerate(query_interval_table):
        d1_left = int(query_interval[0])
        d1_right = int(query_interval[1])
        d2_left = int(query_interval[2])
        d2_right = int(query_interval[3])
        d3_left = int(query_interval[4])
        d3_right = int(query_interval[5])
        d4_left = int(query_interval[6])
        d4_right = int(query_interval[7])
        d5_left = int(query_interval[8])
        d5_right = int(query_interval[9])
        real_frequency_value = user_dataset[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right, d4_left:d4_right, d5_left:d5_right].sum()/user_dataset_size

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

        #
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
            # print(abs(diff_estimated_frequency_value).sum())
            temp_value_vector = estimated_frequency_value_vector.copy()



        errList[i] = real_frequency_value - estimated_frequency_value_vector[0]
        # print(real_frequency_value, estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def theta_calculation(ahead_tree_height, epsilon, user_scale, branch, data_dimension):
    domain_combination_number = data_dimension * (data_dimension-1) / 2
    user_scale_in_each_layer = user_scale / (ahead_tree_height*domain_combination_number)
    varience_of_oue = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((math.prod(branch) + 1) * varience_of_oue)


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


        # 逐个属性进行操作
        for m in range(len(domain_size)):
            layer_frequency_distribution_index_list = []

            # 采集属性位置
            for n, domain_combination in enumerate(domain_combination_list):
                if m in domain_combination:
                    layer_frequency_distribution_index_list.append(n)

            # 对矩阵内频率进行求和
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
        # print(ahead_forest_frequency_distribution_list)
        interval_length = interval_length / branch

    ahead_forest_frequency_distribution_list_transfer_to_ahead_forest = []
    for v in range(len(domain_combination_list)):
        ahead_tree_list = []

        for g in range(ahead_tree_height):
            ahead_tree_list.append(ahead_forest_frequency_distribution_list[g][v])

        ahead_forest_frequency_distribution_list_transfer_to_ahead_forest.append(ahead_tree_list)

    return ahead_forest_frequency_distribution_list_transfer_to_ahead_forest


def main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name, data_dimension):
    repeat = 0
    MSEDict = {'rand': []}

    while repeat < repeat_time:
        ahead_forest = []
        # 用户划分
        start_time = time.time()
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height, domain_size, data_dimension)
        end_time = time.time()
        print('用户划分', end_time - start_time)

        tree_id = 0
        domain_combination_list = []
        # 初始化树结构，设置根结点
        for i in range(data_dimension-1):
            for j in range(i+1, data_dimension):
                domain_combination_list.append([i, j])
                ahead_tree = Tree()
                ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size[0], 0, domain_size[1]])))
                print('初始化树结构，设置根结点')

                # 构建平移向量
                start_time = time.time()
                translation_vector = construct_translation_vector(domain_size, branch)
                end_time = time.time()
                print('构建平移向量', end_time-start_time)

                # 构建树结构
                start_time = time.time()
                ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition[tree_id], epsilon)
                end_time = time.time()
                print('构建树结构', end_time-start_time)

                # ahead_tree后向处理
                start_time = time.time()
                ahead_tree_postprocessing(ahead_tree)
                end_time = time.time()
                # ahead_tree.show(data_property='frequency')
                # ahead_tree.show(data_property='interval')
                print('ahead_tree后向处理', end_time-start_time)

                ahead_forest.append(ahead_tree)
                tree_id += 1
        # print(ahead_forest)

        # 恢复完整的树结构
        for k, ahead_tree in enumerate(ahead_forest):
            restore_the_complete_tree_structure(ahead_tree, branch, ahead_tree_height)
            # print(k, ahead_tree.depth())
            # ahead_tree.show(data_property='frequency')
            # ahead_tree.show(data_property='interval')
            # ahead_tree.show(data_property='count')
            # for i in range(ahead_tree.depth()+1):
            #     print(i, ahead_tree.size(level=i))

        # Consistency Step
        ahead_tree_list = ahead_tree_consistency_step(ahead_forest, ahead_tree_height, domain_size, branch, domain_combination_list)
        print(ahead_tree_list[1][1].shape)

        # for i in range(ahead_tree.depth()):
        #     print(i, ahead_tree.size(level=i))
        # # ahead_tree回答查询
        start_time = time.time()
        ahead_tree_query_error_recorder(ahead_forest, ahead_tree_list, user_dataset, query_interval_table, domain_size, MSEDict, domain_combination_list, data_dimension, branch)
        end_time = time.time()
        print('ahead_tree回答查询', end_time-start_time)

        # 记录误差
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv(
            'rand_result/0410MSE_weight_updated_ahead_branch{}-{}-{}-{}-{}-{}-{}.csv'.format(branch[0],
                                                                                                   data_name,
                                                                                                   data_size_name,
                                                                                                   domain_name,
                                                                                                   epsilon,
                                                                                                   SNR, repeat_time))
        repeat += 1
        print(epsilon, data_name, repeat)


class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval


if __name__ == "__main__":

    repeat_time = 1
    epsilon = 1
    domain_size = [2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6]
    branch = np.array([2, 2, 2, 2, 2])
    data_dimension = 5

    data_name = '5dim_Laplace08'
    data_size_name = 'Set_10_6'
    domain_name = 'Domain_6_6_6_6_6'

    ahead_tree_height = int(math.log(max(domain_size), max(branch)))
    data_path = './dataset/{}-{}-{}-Data.txt'.format(data_name, data_size_name, domain_name)
    query_path = './querytable/Rand_QueryTable_Domain_6_6_6_6_6.npy'
    user_dataset = user_record(data_path, domain_size)
    query_interval_table = np.load(query_path)
    print(query_interval_table.shape)
    SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch[:2], data_dimension)
    main_fun(repeat_time, domain_size[:2], branch[:2], ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name, data_dimension)

'''
    # repeat_time = 20
    # epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    # # data_name_list = ['Five_Laplace01', 'Five_Laplace02', 'Five_Laplace03', 'Five_Laplace04', 'Five_Laplace05', 'Five_Laplace06', 'Five_Laplace07', 'Five_Laplace08', 'Five_Laplace09',
    # #                   'Five_Normal01', 'Five_Normal02', 'Five_Normal03', 'Five_Normal04', 'Five_Normal05', 'Five_Normal06', 'Five_Normal07', 'Five_Normal08', 'Five_Normal09']
    # data_name_list = ['Five_Laplace08', 'Five_Normal08']
    # data_size_name_list = ['Set_10_6', 'Set_10_7']
    # data_size_List = [10 ** 6, 10 ** 7]

    repeat_time = 20
    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    data_name_list = ['BlackFriday', 'Financial', 'Loan', 'Salaries']
    # epsilon_list = [1]
    # data_name_list = ['BlackFriday']
    data_size_name_list = ['Ori']
    domain_name_list = ['Domain6_Attribute5']
    domain_size_list = [[2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6]]

    branch = np.array([2, 2, 2, 2, 2])

    data_dimension = 5


    p = Pool(processes=8)
    for i, data_name in enumerate(data_name_list):
        for j, epsilon in enumerate(epsilon_list):
            for k, domain_name in enumerate(domain_name_list):
                for m, data_size_name in enumerate(data_size_name_list):
                    ahead_tree_height = int(math.log(max(domain_size_list[k]), max(branch)))
                    data_path = './datasets/data/{}-{}-{}-Data.txt'.format(data_name, data_size_name, domain_name)
                    query_path = './query_table/Rand_QueryTable_Domain_6_6_6_6_6.npy'
                    user_dataset = user_record(data_path, domain_size_list[k])
                    query_interval_table = np.load(query_path)
                    print(query_interval_table.shape)
                    SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch[:2], data_dimension)

                    p.apply_async(main_fun, args=(repeat_time, domain_size_list[k][:2], branch[:2], ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name, data_dimension))
                    # main_fun(repeat_time, domain_size_list[k][:2], branch[:2], ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name, data_dimension)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
'''
