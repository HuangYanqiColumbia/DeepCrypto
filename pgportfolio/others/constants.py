from hyperopt import hp

pooling_size = hp.choice("Pooling/kernel_size", [2, 5, 10, 20])

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
			"activation_function": "relu", 
			"filter_shape": [1, 5], 
			"weight_decay": hp.loguniform("ConvLayer/weight_decay", -2, 9), 
			"regularizer": "L1"
		},
		{
			"activation_function": "relu", 
			"type": "EIIE_Dense", 
			"filter_number": 20, 
			"weight_decay": hp.loguniform("EIIE_Dense/weight_decay", -2, 9), 
			"regularizer": "L1"
        },
        {
            "type": "Pooling", 
            "kernel_size": [1,1,pooling_size,1], 
            "strides": [1,1,pooling_size,1]
        },  		
		{
			"type": "EIIE_Output", 
			"weight_decay": hp.loguniform("EIIE_Output/weight_decay", -2, 9), 
			"regularizer": "L1"
		}
	],
    "training": {
        "batch_size": 50,
        "buffer_biased": 5e-04,
        "decay_rate": 1,
        "decay_steps": 50000,
        "learning_rate": hp.loguniform("learning_rate", -5, 2),
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