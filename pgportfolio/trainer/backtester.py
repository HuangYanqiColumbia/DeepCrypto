# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:35:01 2019

@author: Huang Yanqi
"""

import numpy as np
import pandas as pd
from pgportfolio.learning_entities.actor import ActorNetwork
from pgportfolio.data_manager.replay_buffer import ReplayBuffer


class Backtester():
    def __init__(self, sess, config, env, actor: ActorNetwork, num: int=0):
        self._sess = sess
        self._config = config
        self._env = env
        self._actor = actor
        self._pvm = np.array([[1. for _ in range(self._actor.a_dim)]])
        self._num = num
    
    @staticmethod
    def _process_state_batch(s_batch):
        return np.array([np.array([_[0] for _ in s_batch]),
                         s_batch.shape[0]])
    
    def backtest(self):
        self._prepare()        
        s, last_s = self._env.reset(self._env._config)
        self._ep_reward = 0
        self._ep_reward_previous = 0
        self._alphas = np.zeros((0,1+self._actor.a_dim))
        self._rewards = np.zeros((0,2))   
        for j in range(int(self._config["training"]['max_episode_len'])):
            s, last_s = self._evolve(s, last_s, j)
        s_num = "{:04d}".format(self._num)
        result_dir = f'agents/agent_{s_num}/results'
        pd.DataFrame(self._rewards,
                     columns=("time","reward"))\
                     .set_index("time")\
                     .to_csv(result_dir+'/rewards.csv')
        assets = self._env._data_matrices.major_axis
        pd.DataFrame(self._alphas,
                     columns=("time",)+tuple(assets))\
                     .set_index("time")\
                     .to_csv(result_dir+'/alphas.csv')        
    
    def _prepare(self):
        self._replay_buffer = ReplayBuffer(int(self._config["training"]['buffer_size']), 
                                     self._config, int(self._config["training"]['random_seed']))    
    
    def _evolve(self, s, last_s, j):
        a = self._actor.predict([s[0][None,:],1,self._pvm[[-1,]]])  
        s2, close, t = self._env.step(a[0], time_index=True)
        self._alphas = np.append(self._alphas,(np.append([t],np.reshape(a,(self._actor.a_dim,))))[None,:], axis=0) # a*ret.shift(-1) = rewards.shift(-1)
        self._rewards = np.append(self._rewards, [[t,self._env.reward]], axis=0) #rewards.index: realized date
        if len(self._pvm)>=3:
            self._replay_buffer.add(last_s, close, len(self._pvm)-3)
        if self._replay_buffer.size() > int(self._config["training"]["batch_size"]):
            for k in range(int(self._config["training"]["training_times"])):
                s_batch, close_batch, stamp_batch = \
                    self._replay_buffer.sample_batch(int(self._config["training"]["batch_size"]))
                s_batch = self._process_state_batch(s_batch)
                a_batch = self._pvm[stamp_batch]
                self._actor.train(s_batch, close_batch, a_batch)
        self._ep_reward += self._env.reward
        last_s = s
        s = s2
        if j%100==0 and j>0:
            print('episode: {:d} | average reward: {:f}'
                  .format(j, (self._ep_reward-self._ep_reward_previous)/100))
            self._ep_reward_previous = self._ep_reward
        return s, last_s