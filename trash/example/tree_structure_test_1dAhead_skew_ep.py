import treelib
import numpy as np
import math
import freqoracle
from treelib import Tree, Node
import errormetric
import pandas as pd
from multiprocessing import Pool

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
    for i in range(branch):
        translation_vector.append(np.array(
            [i * domain_size // branch, i * domain_size // branch]))
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
                    TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch + node.data.interval[j]
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

    # 这里写一个translationVector去重操作


def user_record(data_path, domain_size):
    dataset = np.load(data_path)
    dataset = np.array(dataset, np.int64)
    # print(dataset)
    # dataset = np.array(dataset['dataset'])
    user_histogram = np.zeros(domain_size, dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[item] += 1
    return user_histogram


def user_record_partition(data_path, ahead_tree_height, domain_size):
    dataset = np.load(data_path)
    dataset = np.array(dataset, np.int64)
    # print(dataset)
    # dataset = np.array(dataset['dataset'])
    user_sample_id = np.random.randint(0, ahead_tree_height, len(dataset)).reshape(len(dataset), 1)  # user sample list
    user_histogram = np.zeros((ahead_tree_height, domain_size), dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[user_sample_id[k], item] += 1
    return user_histogram


def ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition, epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # ahead_tree结构更新
        ahead_tree_update(ahead_tree, tree_height, SNR, branch, translation_vector, layer_index)
        # 更新平移向量
        translation_vector[:] = translation_vector[:] // np.array([branch, branch])
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
        user_record_list.append(user_dataset[d1_left:d1_right].sum())

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
            node.data.count = 1/coeff0


def ahead_tree_answer_query(ahead_tree, query_interval, domain_size):
    estimated_frequency_value = 0
    # 设置查询区域
    query_interval_temp = np.zeros(domain_size)
    d1_left = int(query_interval[0])
    d1_right = int(query_interval[1])
    query_interval_temp[d1_left:d1_right] = 1

    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])

        if query_interval_temp.sum() and ahead_tree.children(node.identifier) != [] and query_interval_temp[d1_left:d1_right].sum() == (d1_right-d1_left):
            estimated_frequency_value = estimated_frequency_value + node.data.frequency
            query_interval_temp[d1_left:d1_right] = 0
            continue

        if query_interval_temp.sum() and ahead_tree.children(node.identifier) == []:
            coeff = query_interval_temp[d1_left:d1_right].sum()/(d1_right-d1_left)
            estimated_frequency_value = estimated_frequency_value + coeff * node.data.frequency
            query_interval_temp[d1_left:d1_right] = 0

    return estimated_frequency_value, query_interval_temp.sum()


def ahead_tree_query_error_recorder(ahead_tree, user_dataset, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    user_dataset_size = user_dataset.sum()
    for i, query_interval in enumerate(query_interval_table):
        d1_left = int(query_interval[0])
        d1_right = int(query_interval[1])
        real_frequency_value = user_dataset[d1_left:d1_right].sum()/user_dataset_size
        estimated_frequency_value, query_interval_sum = ahead_tree_answer_query(ahead_tree, query_interval, domain_size)
        errList[i] = real_frequency_value - estimated_frequency_value
        # print(real_frequency_value, estimated_frequency_value)
        # print(repeat, i)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def theta_calculation(ahead_tree_height, epsilon, user_scale, branch):
    user_scale_in_each_layer = user_scale / ahead_tree_height
    varience_of_OUE = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((branch + 1) * varience_of_OUE)


def main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name):
    repeat = 0
    MSEDict = {'rand': []}
    while repeat < repeat_time:

        # 初始化树结构，设置根结点
        ahead_tree = Tree()
        ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size])))
        # print('初始化树结构，设置根结点')

        # 构建平移向量
        translation_vector = construct_translation_vector(domain_size, branch)
        # print('构建平移向量')

        # 用户划分
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height, domain_size)
        # print('用户划分')

        # 构建树结构
        ahead_tree_construction(ahead_tree, ahead_tree_height, SNR, branch, translation_vector, user_dataset_partition, epsilon)
        # print('构建树结构')

        # ahead_tree后向处理
        ahead_tree_postprocessing(ahead_tree)
        # ahead_tree.show(data_property='frequency')
        # ahead_tree.show(data_property='interval')
        # print('ahead_tree后向处理')

        # ahead_tree回答查询
        ahead_tree_query_error_recorder(ahead_tree, user_dataset, query_interval_table, domain_size, MSEDict)
        # print('ahead_tree回答查询')

        # 记录误差
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv(
            'rand_result/0328MSE_ARQ_normsub_optimize_nodes_branch{}-{}-{}-{}-{}-{}-{}.csv'.format(branch,
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

    epsilon_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    data_name_list = ['Normal_Skew00', 'Normal_Skew01', 'Normal_Skew02', 'Normal_Skew03', 'Normal_Skew04', 'Normal_Skew05','Normal_Skew06', 'Normal_Skew07', 'Normal_Skew08', 'Normal_Skew09']
    # epsilon_list = [0.01, 0.02]
    # data_name_list = ['Normal_Skew00', 'Normal_Skew01']
    data_size_name_list = ['Set_10_6', 'Set_10_7']
    domain_name = 'Domain_10'
    domain_size = 2 ** 10
    branch = 2
    ahead_tree_height = int(math.log(domain_size, branch))
    repeat_time = 20

    p = Pool(processes=18)
    for i, data_name in enumerate(data_name_list):
        for j, epsilon in enumerate(epsilon_list):
            for k, data_size_name in enumerate(data_size_name_list):
                data_path = './datasets/data/{}-{}-{}-Data.npy'.format(data_name, data_size_name, domain_name)
                query_path = './query_table/Rand_QueryTable_Domain_2_10.npy'
                user_dataset = user_record(data_path, domain_size)
                query_interval_table = np.load(query_path)
                SNR = theta_calculation(ahead_tree_height, epsilon, user_dataset.sum(), branch)

                p.apply_async(main_fun, args=(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name))
                # main_fun(repeat_time, domain_size, branch, ahead_tree_height, SNR, user_dataset, query_interval_table, epsilon)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')





