# DeepCrypto
## A hyperOpt framework to train deep network for trading in crypto assets

#### Introduction
This flesh of the project is in the folder, ```pgportfolio```. However, to start off the project, you also need to supply data in the ```data_base``` folder. To start off the hyper-parameter tuning, simply run ```python main.py --mode=hyperopt```. 

#### Asynchronous Hyperparameter Tuning
Asynchronous hyperopt may be enabled. You need to start a mongodb server by ```mongod --dbpath ./data_base/mongodb --port 1234```. Within the true working directory, ```./mongo_workers/```, you may need to spawn several mongo-workers by ```PYTHONPATH=.. hyperopt-mongo-worker --mongo=localhost:1234/my_db --poll-interval=0.1```. You should also ask hyperopt to assign tasks by ```python main.py --mode=hyperopt --train_option=mongo --round=1```

#### Data Feeding
The data should be fed in the format of parquet.gzip, which is efficient in terms of space and speed of reading/writing. The path to a specific date should look something like ```./data_base/data/2019/20190103.parquet.gzip```. You can download the data from Poloniex. There's a function in ```data_manager``` that helps you write panel data in pandas to the path described above. The functionality will be formalized in the future as a part of those provided by the ```main.py```.

#### Performance
A preliminary training of the agents generate agent_0003 as the best performer. The pnl plot looks like
![PNL for BackTesting by agent_0003](misc/pics/pnls.png)
