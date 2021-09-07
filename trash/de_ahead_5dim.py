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
            for k in range(branch[2]):
                for m in range(branch[3]):
                    for n in range(branch[4]):
                        translation_vector.append(np.array(
                            [i * domain_size[0] // branch[0], i * domain_size[0] // branch[0],
                             j * domain_size[1] // branch[1], j * domain_size[1] // branch[1],
                             k * domain_size[2] // branch[2], k * domain_size[2] // branch[2],
                             m * domain_size[3] // branch[3], m * domain_size[3] // branch[3],
                             n * domain_size[4] // branch[4], n * domain_size[4] // branch[4]]))
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
                    TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch[j // 2] + \
                                       node.data.interval[j]
                else:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = node.data.interval[j + 1]

            for item1 in translation_vector:
                node_name = str(tree_height) + str(layer_index)
                node_frequency = 0
                node_divide_flag = True
                node_count = 0
                node_interval = TempItem0 + item1
                ahead_tree.create_node(node_name, node_name, parent=node.identifier,
                                       data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
                layer_index += 1


def user_record(data_path, domain_size):
    dataset = np.loadtxt(data_path, np.int32)
    # print(dataset)
    # dataset = np.array(dataset['dataset'])
    user_histogram = np.zeros(domain_size, dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[item[0], item[1], item[2]] += 1
    return user_histogram


def user_record_partition(data_path, ahead_tree_height, domain_size):
    dataset = np.loadtxt(data_path, np.int32)
    # print(dataset)
    # dataset = np.array(dataset['dataset'])
    user_histogram = np.zeros(
        (ahead_tree_height, domain_size[0], domain_size[1], domain_size[2], domain_size[3], domain_size[4]),
        dtype=np.int32)
    for k, item in enumerate(dataset):
        user_sample_id = np.random.randint(0, ahead_tree_height)
        user_histogram[user_sample_id, item[0], item[1], item[2], item[3], item[4]] += 1
    return user_histogram


def ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition,
                            epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # ahead_tree结构更新
        ahead_tree_update(ahead_tree, tree_height, SNR, branch, translation_vector, layer_index)
        # 更新平移向量
        translation_vector[:] = translation_vector[:] // np.array(
            [branch[0], branch[0], branch[1], branch[1], branch[2], branch[2], branch[3], branch[3], branch[4],
             branch[4]])
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
        d3_left = int(node.data.interval[4])
        d3_right = int(node.data.interval[5])
        d4_left = int(node.data.interval[6])
        d4_right = int(node.data.interval[7])
        d5_left = int(node.data.interval[8])
        d5_right = int(node.data.interval[9])

        user_record_list.append(user_dataset[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right, d4_left:d4_right,
                                d5_left:d5_right].sum())

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
            numerator = 1 / node.data.count
            children_frequency_sum = 0
            for j, child_node in enumerate(ahead_tree.children(node.identifier)):
                numerator += 1 / child_node.data.count
                children_frequency_sum += child_node.data.frequency

            denominator = numerator + 1
            coeff0 = numerator / denominator
            coeff1 = 1 - coeff0

            node.data.frequency = coeff0 * node.data.frequency + coeff1 * children_frequency_sum
            node.data.count = 1 / coeff0


def ahead_tree_answer_query(ahead_tree, query_interval, domain_size):
    estimated_frequency_value = 0

    # 设置查询区域
    query_interval_temp = np.zeros(domain_size, np.int8)
    query_d1_left = int(query_interval[0])
    query_d1_right = int(query_interval[1])
    query_d2_left = int(query_interval[2])
    query_d2_right = int(query_interval[3])
    query_d3_left = int(query_interval[4])
    query_d3_right = int(query_interval[5])
    query_d4_left = int(query_interval[6])
    query_d4_right = int(query_interval[7])
    query_d5_left = int(query_interval[8])
    query_d5_right = int(query_interval[9])
    query_interval_temp[query_d1_left:query_d1_right, query_d2_left:query_d2_right, query_d3_left:query_d3_right,
    query_d4_left:query_d4_right, query_d5_left:query_d5_right] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):

        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        d2_left = int(node.data.interval[2])
        d2_right = int(node.data.interval[3])
        d3_left = int(node.data.interval[4])
        d3_right = int(node.data.interval[5])
        d4_left = int(node.data.interval[6])
        d4_right = int(node.data.interval[7])
        d5_left = int(node.data.interval[8])
        d5_right = int(node.data.interval[9])

        query_interval_temp_sum = query_interval_temp[query_d1_left:query_d1_right, query_d2_left:query_d2_right,
                                  query_d3_left:query_d3_right, query_d4_left:query_d4_right,
                                  query_d5_left:query_d5_right].sum()

        query_interval_area = (d5_right - d5_left) * (d4_right - d4_left) * (d3_right - d3_left) * (
                    d2_right - d2_left) * (d1_right - d1_left)

        if ahead_tree.children(node.identifier) != [] and query_interval_temp_sum == query_interval_area:
            estimated_frequency_value = estimated_frequency_value + node.data.frequency
            query_interval_temp[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right, d4_left:d4_right,
            d5_left:d5_right] = 0
            continue

        if ahead_tree.children(node.identifier) == []:
            coeff = query_interval_temp_sum / query_interval_area
            estimated_frequency_value = estimated_frequency_value + coeff * node.data.frequency
            query_interval_temp[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right, d4_left:d4_right,
            d5_left:d5_right] = 0

    return estimated_frequency_value


def ahead_tree_query_error_recorder(ahead_tree, user_dataset, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    user_dataset_size = user_dataset.sum()
    for i, query_interval in enumerate(query_interval_table):
        print(i, query_interval)
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
        real_frequency_value = user_dataset[d1_left:d1_right, d2_left:d2_right, d3_left:d3_right, d4_left:d4_right,
                               d5_left:d5_right].sum() / user_dataset_size
        estimated_frequency_value = ahead_tree_answer_query(ahead_tree, query_interval, domain_size)
        errList[i] = real_frequency_value - estimated_frequency_value
        # print(real_frequency_value, estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def theta_calculation(ahead_tree_height, epsilon, user_scale, branch):
    user_scale_in_each_layer = user_scale / ahead_tree_height
    varience_of_oue = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((math.prod(branch) + 1) * varience_of_oue)


def main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon,
             data_path, data_name, data_size_name, domain_name):
    repeat = 0
    MSEDict = {'rand': []}
    while repeat < repeat_time:

        # 初始化树结构，设置根结点
        ahead_tree = Tree()
        ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array(
            [0, domain_size[0], 0, domain_size[1], 0, domain_size[2], 0, domain_size[3], 0, domain_size[4]])))
        print('初始化树结构，设置根结点')

        # 构建平移向量
        start_time = time.time()
        translation_vector = construct_translation_vector(domain_size, branch)
        end_time = time.time()
        print('构建平移向量', end_time - start_time)

        # 用户划分
        start_time = time.time()
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height, domain_size)
        end_time = time.time()
        print('用户划分', end_time - start_time)

        # 构建树结构
        start_time = time.time()
        ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition,
                                epsilon)
        end_time = time.time()
        print('构建树结构', end_time - start_time)

        # ahead_tree后向处理
        start_time = time.time()
        ahead_tree_postprocessing(ahead_tree)
        end_time = time.time()
        # ahead_tree.show(data_property='frequency')
        # ahead_tree.show(data_property='interval')
        print('ahead_tree后向处理', end_time - start_time)

        for i in range(ahead_tree.depth() + 2):
            print(i, ahead_tree.size(level=i))
        # ahead_tree回答查询
        start_time = time.time()
        ahead_tree_query_error_recorder(ahead_tree, user_dataset, query_interval_table, domain_size, MSEDict)
        end_time = time.time()
        print('ahead_tree回答查询', end_time - start_time)

        # 记录误差
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv(
            'rand_result/0404MSE_standard_ahead_branch{}-{}-{}-{}-{}-{}-{}.csv'.format(branch[0],
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
    SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch)
    
    main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name)


'''
    # epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    data_name_list = ['Five_Laplace08']

    epsilon_list = [1.1]
    repeat_time = 2
    # data_name_list = ['Three_Laplace01', 'Three_Laplace02', 'Three_Laplace03', 'Three_Laplace04', 'Three_Laplace05', 'Three_Laplace06', 'Three_Laplace07']
    data_size_name_list = ['Set_10_6']
    domain_name_list = ['Domain_6_6_6_6_6']
    domain_size_list = [[2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6, 2 ** 6]]

    branch = [2, 2, 2, 2, 2]

    # p = Pool(processes=8)
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
                    SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch)

                    # p.apply_async(main_fun, args=(repeat_time, domain_size_list[k], branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name))
                    main_fun(repeat_time, domain_size_list[k], branch, ahead_tree_height, SNR, user_dataset,
                             query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name)
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')


    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    # epsilon_list = [0.1, 0.3]
    data_name_list = ['Three_Laplace08', 'Three_Normal08']
    # data_size_name_list = ['Set_10_7']
    data_size_name = 'Set_10_7'
    domain_name_list = ['Domain_8_8_8', 'Domain_10_10_10']
    domain_size_list = [[2 ** 8, 2 ** 8, 2 ** 8], [2 ** 10, 2 ** 10, 2 ** 10]]
    # domain_name = 'Domain_8_8_8'
    # domain_size = [2 ** 8, 2 ** 8, 2 ** 8]
    branch = [2, 2, 2]

    repeat_time = 20

    p = Pool(processes=32)
    for i, data_name in enumerate(data_name_list):
        for j, epsilon in enumerate(epsilon_list):
            for k, domain_name in enumerate(domain_name_list):

                ahead_tree_height = int(math.log(max(domain_size_list[k]), max(branch)))
                data_path = './datasets/data/{}-{}-{}-Data.npy'.format(data_name, data_size_name, domain_name)
                query_path = './query_table/Rand_QueryTable_{}.npy'.format(domain_name_list[k])
                user_dataset = user_record(data_path, domain_size_list[k])
                query_interval_table = np.load(query_path)
                SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch)

                p.apply_async(main_fun, args=(repeat_time, domain_size_list[k], branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name))
                # main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
'''


