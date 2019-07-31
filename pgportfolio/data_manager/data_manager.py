import os
import pandas as pd
from datetime import datetime

class DataManager():
    def __init__(self, dbase_path:str="./data_base/data", start:datetime=datetime(2017,12,31), end:datetime=datetime(2019, 4, 19)):
        self._dbase_path = dbase_path
        self._start = start
        self._end = end # inclusive
        self._data_matrices = None
        self._items = ["close", "high", "low"]

    def load_data(self):
        dfs = dict()
        for item in self._items:
            l = []
            for year in os.listdir(self._dbase_path):
                for date in os.listdir(f"{self._dbase_path}/{year}"):  
                    if datetime.strptime(date, "%Y%m%d")>self._end or datetime.strptime(date, "%Y%m%d")<self._start:
                        continue            
                    l.append(
                        pd.read_parquet(
                            f"{self._dbase_path}/{year}/{date}/"
                            f"{item}.parquet.gzip", engine="fastparquet"
                        )
                    )
            dfs[item] = pd.concat(l, axis=0)
        self._data_matrices = pd.Panel(dfs).transpose(0, 2, 1)
    
    @property
    def data_matrices(self):
        return self._data_matrices
    
    @property
    def data(self):
        return self._data_matrices.values