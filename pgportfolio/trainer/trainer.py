# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:18:39 2019

@author: Huang Yanqi
"""

import numpy as np
import tensorflow as tf
from pgportfolio.learning_entities.actor import ActorNetwork
from pgportfolio.data_manager.replay_buffer import ReplayBuffer

import logging 
logger = logging.getLogger('trainer')
hdlr = logging.FileHandler("./misc/logs/trainer.log")
formatter = logging.Formatter('%(name)s - %(levelname)s\n%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)
logger.propagate = False

class Trainer():
    def __init__(self, sess, config, global_step: int\
                 , env, actor: ActorNetwork, saver, pn: str, restored = False):
        self._sess = sess
        self._config = config
        self._global_step = global_step
        self._env = env
        self._actor = actor
        self._saver = saver
        self._restored = restored
        self._pn = pn
        self.a_dim = self._config["input"]["asset_num"]
        self._prev_ep_reward = None
    
    @staticmethod
    def _build_summaries():
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)    
        summary_vars = [episode_reward,]
        summary_ops = tf.summary.merge_all()
    
        return summary_ops, summary_vars
    
    @staticmethod
    def _process_state_batch(s_batch):
        return np.array([np.array([_[0] for _ in s_batch]),
                         s_batch.shape[0]])
    
    def train(self):
        summary_ops, summary_vars, writer, increment_global_step_op = self._prepare()        
        for i in range(int(self._config["training"]["max_episodes"])):
            self._replay_buffer = ReplayBuffer(int(self._config["training"]['buffer_size']), 
                                     self._config, int(self._config["training"]['random_seed']))
            self._pvm = np.array([[1 for _ in range(self.a_dim)]])		
            s, last_s = self._env.reset(self._env._config)
            self._ep_reward = 0
            self._ep_reward_previous = 0
            for j in range(int(self._config["training"]['max_episode_len'] * 0.8)):
                s, last_s = self._evolve(s, last_s, j)
            will_stop = self._summarize(writer, summary_ops, summary_vars, increment_global_step_op, i)
            if will_stop:
                break
    
    def _prepare(self):
        summary_ops, summary_vars = self._build_summaries()
        increment_global_step_op = tf.assign(self._global_step, self._global_step+1)
        if self._restored:
            print("*"*40)
            print("session restored from global step %d!" %(self._sess.run(self._global_step),))
            print("*"*40)
        if not self._restored:
            self._sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self._pn+"/summary", self._sess.graph)
        return summary_ops, summary_vars, writer, increment_global_step_op
    
    def _summarize(self, writer, summary_ops, summary_vars, increment_global_step_op, i):
        will_stop=False
        summary_str = self._sess.run(summary_ops, feed_dict={
            summary_vars[0]: self._ep_reward
        })
        if self._prev_ep_reward is not None and self._ep_reward - self._prev_ep_reward<1e-01:
            will_stop=True
        self._sess.run(increment_global_step_op)
        writer.add_summary(summary_str, self._sess.run(self._global_step))
        writer.flush()
        save_path = self._saver.save(self._sess, self._pn+"/session/netfile")
        print("agent trained saved to %s" %(save_path,))
        print('| Reward: {:.4f} | Episode: {:d}'.format(self._ep_reward, self._sess.run(self._global_step)))
        self._prev_ep_reward = self._ep_reward
        return will_stop
    
    def _evolve(self, s, last_s, j):
        a = self._actor.predict([s[0][None,:],1,self._pvm[[-1,]]])
        if logger.level <= logging.DEBUG:
            logger.debug(
                a[0, :5]
            )
        self._pvm = np.append(self._pvm, a, axis=0)
        s2, close = self._env.step(a[0])
        if len(self._pvm)>=3:
            self._replay_buffer.add(last_s, close, len(self._pvm)-3)
        if self._replay_buffer.size() > int(self._config["training"]["batch_size"]):
            for k in range(int(self._config["training"]["training_times"])):
                s_batch, close_batch, stamp_batch = \
                    self._replay_buffer.sample_batch(int(self._config["training"]["batch_size"]))
                s_batch = self._process_state_batch(s_batch)
                a_batch = self._pvm[stamp_batch]
                self._actor.train(s_batch, close_batch, a_batch)
                self._pvm[stamp_batch+1] = self._actor.predict([s_batch[0], s_batch[1], a_batch])
        self._ep_reward += self._env.reward
        last_s = s
        s = s2
        if j%100==0 and j>0:
            print('episode: {:d} | average reward: {:f}'
                  .format(j, (self._ep_reward-self._ep_reward_previous)/100))
            self._ep_reward_previous = self._ep_reward
        return s, last_s