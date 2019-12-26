# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:00:27 2019

@author: Huang Yanqi
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pgportfolio.data_manager.data_manager import DataManager

class Env():
    def __init__(self, config):
        self._config = config
        self.set_config()
        self._data_manager = None
        self._data_matrices = None 
        self._data = None
        self.load_data()
        self._curr_ind = self._window_size-1
        self.reward = None
        
    def load_data(self):
        self._data_manager = DataManager(start=self._start_date, end=self._end_date)
        self._data_manager.load_data()
        self._data_matrices = self._data_manager.data_matrices # (features, assets, times) 
        self._data = self._data_matrices.values  
    
    def reset(self, config):
        self._curr_ind = self._window_size-1
        self.weight = np.array([1,])
        return self.step(self.weight)[0], [self._data[[1,],:,0:self._window_size], np.array([1,])]
    
    def set_config(self):
        self.set_window_size()
        self.set_dates()
    
    def set_window_size(self):
        config = self._config.copy()
        input_config = config["input"]
        self._window_size = input_config["window_size"]

    def set_dates(self):
        config = self._config.copy()
        self._end_date = datetime.strptime(config["input"]["end_date"],"%Y/%m/%d")
        self._start_date = datetime.strptime(config["input"]["start_date"],"%Y/%m/%d")

    def step(self, action, time_index=False):
        self._curr_ind += 1
        s2 = self._data[:,:,self._curr_ind-self._window_size+1:self._curr_ind+1]
        close = self._data[0,:,self._curr_ind-self._window_size:self._curr_ind+1].T
        retrn = (self._data[0,:,self._curr_ind] - self._data[0,:,self._curr_ind-1]) / self._data[0,:,self._curr_ind-1]
        pnl = np.nansum(self.weight * retrn)
        commission = np.nansum(np.abs(self.weight * (retrn + 1) - action) * self._config["training"]["trading_consumption"])
        self.reward = pnl - commission
        self.weight = action
        if time_index:
            return ([s2,], close,
                    self._data_matrices.minor_axis[self._curr_ind])
        return [s2,], close
