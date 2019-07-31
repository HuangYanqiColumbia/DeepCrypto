# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:00:27 2019

@author: Huang Yanqi
"""

import os
import json
import logging
import tflearn
import pandas as pd
import tensorflow as tf
from pathlib import Path
from pgportfolio.learning_entities.env import Env
from pgportfolio.learning_entities.actor import ActorNetwork
from pgportfolio.trainer.trainer import Trainer

def _make_and_copy(num: int):
    from shutil import copyfile
    s = "{:04d}".format(num)
    pn = f'agents/agent_{s}'
    for dire in ['summary', 'session', 'results']:
        p = Path(pn+'/'+dire)
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            logging.warning('%s not found and created' %(pn+'/'+dire))
    if not Path(pn+'/net_config.json').is_file():
        copyfile('agents/sample.json', pn+'/net_config.json')
        logging.warning('net_config.json copied from sample')        
    return pn
    
        
def train(num):
    with tf.Session() as sess:
        tflearn.is_training(True, sess)
        pn = _make_and_copy(num)
        with open(pn+"/net_config.json") as file:
            config = json.loads(file.read())

        env = Env(config)
        actor = ActorNetwork(sess, config)
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        saver = tf.train.Saver()
        restored = False
        if os.path.isfile(pn+'/session/checkpoint'):
            saver.restore(sess, pn+'/session/netfile')
            restored=True
        trainer = Trainer(sess, config, global_step, env, actor, saver, pn, restored)
        trainer.train()
        save_path = saver.save(sess, pn+'/session/netfile')
        print("agent trained saved to %s" %(save_path,))
