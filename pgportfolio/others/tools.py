import numpy as np
import pandas as pd

def serialize(dic: dict):
    def func(dic, k):
        if type(dic[k])==np.int32:
            dic[k] = int(dic[k])
    for k, v in dic.items():
        if isinstance(v, dict):
            serialize(v)
        else:
            func(dic, k)

def cal_out_of_sample_perf(num):
    num = "{:04d}".format(num)
    reward = pd.read_csv(f'agents/agent_{num}/results/rewards.csv', index_col=0, header=0).reward
    return reward.iloc[-int(reward.shape[0]*0.2):].cumsum(skipna=True).iloc[-1]