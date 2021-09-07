import numpy as np
import copy

def real_frequency_generation(dataset, data_size, domain_size, data_dimension, query_interval_table):
    user_histogram = np.zeros(domain_size, dtype=np.int32)
    real_frequency_value = np.zeros(len(query_interval_table))
    
    if data_dimension == 1:
        for k, item in enumerate(dataset):
            user_histogram[item] += 1
    else:
        for k, item in enumerate(dataset):
            user_histogram[tuple(item)] += 1
        
    for i, query_interval in enumerate(query_interval_table):
        user_histogram_copy = copy.deepcopy(user_histogram)
        
        for j in range(data_dimension):
            user_histogram_copy = np.swapaxes(user_histogram_copy, 0, j)
            user_histogram_copy = user_histogram_copy[query_interval[2*j]: query_interval[2*j+1]]
            
        real_frequency_value[i] = user_histogram_copy.sum()/data_size
        
    return real_frequency_value