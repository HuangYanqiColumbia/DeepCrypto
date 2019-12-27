import os
import numpy as np
import pandas as pd

def serialize(dic: dict):
    def func(dic, k):
        if (type(dic[k]) == np.int32) or (type(dic[k]) == np.int64):
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

def clear_var_files_for_agent(num: int, root = "."):
    logs_path = f"{root}/misc/logs"
    for file in os.listdir(logs_path):
        if file==".gitkeep":
            continue
        os.remove(f"{logs_path}/{file}")
    num = "{:04d}".format(num)
    agent_path = f"{root}/agents/agent_{num}"
    for file in os.listdir(agent_path):
        if file == "net_config.json":
            continue
        os.remove(f"{agent_path}/{file}")
    
def clear_var_files(root = "."):
    logs_path = f"{root}/misc/logs"
    for file in os.listdir(logs_path):
        if file==".gitkeep":
            continue
        os.remove(f"{logs_path}/{file}")
    agents_path = f"{root}/agents"
    for file in os.listdir(agents_path):
        if file == "sample.json":
            continue
        os.remove(f"{agents_path}/{file}")

def load_minutely_panel(root = "."):
    sub_db_path = f"{root}/data_base/minute_data"
    data = dict()
    for file in os.listdir(sub_db_path):
        if file == ".gitkeep":
            continue
        data[file[:-len(".csv")]] = pd.read_csv(f"{sub_db_path}/{file}", index_col = 0)
    return pd.Panel(data)
        
def backtest_with_lag(alpha, root = ".", lag: int = 1):
    panel = load_minutely_panel(root)
    close = panel.close; del panel
    close = close.shift(- lag).loc[alpha.index]
    retrn = (close.shift(-1) - close) / close
    pnls = (alpha * retrn).sum(axis = 1)
    return pnls
    
        
    
    