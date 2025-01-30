import numpy as np
import pickle
import pandas as pd

def normalize_eps(all_eps):
    # 1. 将all_eps中的所有数据提取到一个一维数组
    all_data = np.concatenate(all_eps)
    # 2. 计算均值和标准差
    mean = np.mean(all_data)
    std = np.std(all_data)
    # 3. 对所有数据进行Z-score标准化
    normalized_data = (all_data - mean) / std
    # 4. 将标准化后的数据重新组织成原始结构
    normalized_eps = []
    index = 0
    for group in all_eps:
        normalized_eps.append(normalized_data[index:index+len(group)])
        index += len(group)
    # for i, group in enumerate(normalized_eps):
    #     print(f"Group {i+1}: {group}")
    return normalized_eps