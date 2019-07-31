# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:48:16 2019

@author: Huang Yanqi
"""
import numpy as np
import tensorflow as tf
from pgportfolio.learning_entities.network import CNN

class ActorNetwork(object):
    def __init__(self, sess, config):
        self.sess = sess
        self._config = config
        self.a_dim = config["input"]["asset_num"]
        self.learning_rate = self._config["training"]["learning_rate"]
        self.tau = self._config["training"]["tau"]
        self.batch_size = self._config["training"]["batch_size"]
        self._window_size = self._config["input"]["window_size"]
        self.create_actor_network()
        self.network_params = tf.trainable_variables()
        self._input_num = self._config["training"]["batch_size"]

        self.close = tf.placeholder(tf.float32,[self._input_num, self._window_size+1, self.a_dim])
        close = self.close
        pnl = self.out * ((close[:,1:,:]-close[:,:-1,:])/close[:,:-1,:])[:,-1,:]
        
        weight = self.inputs[2]
        commission = tf.abs(self.out - weight) * self._config["training"]["trading_consumption"]
        
        returns = pnl - commission
        self.loss = - tf.reduce_mean(tf.reduce_sum(returns,axis=1),axis=0)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                minimize(self.loss)

        self.num_trainable_vars = len(self.network_params)

    def create_actor_network(self):
        self._net = CNN(self._config["input"]["feature_number"],
                        self.a_dim, 
                        self._config["input"]["window_size"],
                        self._config["layers"])
        self.inputs = [self._net.input_tensor, self._net.input_num, self._net.weight]
        self.out = self._net.output

    def train(self, inputs, close, a):
        self.sess.run(self.optimize, feed_dict={
            self.inputs[0]: inputs[0],
            self.inputs[1]: inputs[1],
            self.inputs[2]: a, 
            self.close: close
        })

    def predict(self, inputs):
        assert (not np.isnan(inputs[0]).any())
        assert (not np.isinf(inputs[0]).any())
        assert (not np.isinf(inputs[2]).any())
        assert (not np.isinf(inputs[2]).any())
        
        return self.sess.run(self.out, feed_dict={
            self.inputs[0]: inputs[0],
            self.inputs[1]: inputs[1], 
            self.inputs[2]: inputs[2]
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
