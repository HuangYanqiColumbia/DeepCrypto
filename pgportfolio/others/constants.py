import numpy as np
from hyperopt import hp

_POOLING_SIZE = hp.choice("Pooling/kernel_size", [2, 4, 8, 16])
_BASE10 = lambda x: np.log(10 ** x)
_WEIGHT_DECAY = hp.loguniform("weight_decay", _BASE10(-1), _BASE10(1))

HP_SPACE = {
    "input": {
        "end_date": "2019/04/20",
        "feature_number": 3,
        "start_date": "2017/01/01",
        "window_size": hp.choice("window_size", [10, 50, 100, 200, 400]), 
        "asset_num": 10
    },
    "layers": [
		{
			"type": "ConvLayer", 
			"filter_number": 3, 
			"strides": [1, 1], 
			"padding": "valid", 
			"activation_function": "leaky_relu", 
			"filter_shape": [1, hp.choice("ConvLayer/filter_shape", [2, 4, 8])], 
			"weight_decay": _WEIGHT_DECAY, 
			"regularizer": "L1"
		},
		{
			"activation_function": "leaky_relu", 
			"type": "EIIE_Dense", 
			"filter_number": 20, 
			"weight_decay": _WEIGHT_DECAY,
			"regularizer": "L1"
        },
        {
            "type": "Pooling", 
            "kernel_size": [1,1,_POOLING_SIZE,1], 
            "strides": [1,1,_POOLING_SIZE,1]
        },  		
		{
			"type": "EIIE_Output", 
			"weight_decay": _WEIGHT_DECAY, 
			"regularizer": "L1"
		}
	],
    "training": {
        "batch_size": 50,
        "buffer_biased": 5e-04,
        "decay_rate": 1,
        "decay_steps": 50000,
        "learning_rate": hp.loguniform("learning_rate", _BASE10(-1), _BASE10(1)),
        "tau": 0.01,
        "gamma": 1,
        "steps": 500000,
        "max_episode_len": 22000, 
        "max_episodes": 12,
        "random_seed": 123,
        "buffer_size": 1000000,
        "training_times": 5, 
        "trading_consumption": 2.5e-03, 
    }
}