#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn
import abc

class NeuralNetWork:
    def __init__(self, feature_number, assets_num, window_size, layers):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        self.input_num = tf.placeholder(tf.int32, shape=[]) 
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, assets_num, window_size])
        self.weight = tf.placeholder(tf.float32, shape=[None, assets_num])
        self._window_size = window_size
        self._assets_num = assets_num
        self.output = self._build_network(layers)
    
    @abc.abstractmethod
    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, assets_num, window_size, layers):
        NeuralNetWork.__init__(self, feature_number, assets_num, window_size, layers)

    def _build_network(self, layers):
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network
        for _, layer in enumerate(layers):
            if layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 weights_init='variance_scaling')
            elif layer["type"] == "ConvLayer":
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 self._allint(layer["filter_shape"]),
                                                 self._allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 weights_init='variance_scaling')
            elif layer["type"] == "Pooling":
                """
                network_max = tflearn.layers.conv.max_pool_2d(network\
                                                              , kernel_size = layer["kernel_size"]\
                                                              , strides = layer["strides"])
                network_avg = tflearn.layers.conv.avg_pool_2d(network\
                                                              , kernel_size = layer["kernel_size"]\
                                                              , strides = layer["strides"])  
                network = tf.concat([network_max, network_avg], -1)
                '''
				network_l2 = tf.sqrt(tflearn.layers.conv.avg_pool_2d(tf.square(network)\
                                                                     , kernel_size = layer["kernel_size"]\
                                                                     , strides = layer["strides"]))
                '''
                """
                continue
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                weight = tf.reshape(self.weight, [self.input_num, int(height), 1, 1])
                network = tf.concat([network, weight], axis=3)
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 weights_init='variance_scaling')
                network = network[:, :, 0, 0]
                network = self._batch_normalize_2d(network)
                network = tflearn.activations.softmax(network)
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network

    @staticmethod
    def _batch_normalize_2d(network):
        mean = tf.tile(
            tf.math.reduce_mean(network, axis = 1)[:, None]
            , multiples = (1, tf.shape(network)[1])
        )
        std = tf.tile(
            tf.math.reduce_std(network, axis = 1)[:, None]
            , multiples = (1, tf.shape(network)[1])
        )
        network = (network - mean) / std
        return network

    @staticmethod
    def _allint(l):
        return [int(i) for i in l]

