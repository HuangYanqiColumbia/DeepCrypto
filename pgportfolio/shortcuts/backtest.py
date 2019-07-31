# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:00:27 2019

@author: Huang Yanqi
"""

import json
import tflearn
import pandas as pd
import tensorflow as tf
from pgportfolio.learning_entities.env import Env
from pgportfolio.learning_entities.actor import ActorNetwork
from pgportfolio.trainer.backtester import Backtester


def backtest(num: int):
    with tf.Session() as sess:
        tflearn.is_training(True, sess)
        s = "{:04d}".format(num)
        pn = f'agents/agent_{s}'
        with open(pn+"/net_config.json") as file:
            config = json.loads(file.read())
        env = Env(config)
        actor = ActorNetwork(sess, config)
        saver = tf.train.Saver()
        saver.restore(sess, pn+'/session/netfile')
        backtester = Backtester(sess, config, env, actor, num)
        backtester.backtest() 