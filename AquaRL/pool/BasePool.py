import abc
import logging
import numpy as np
import tensorflow as tf
from AquaRL.args import EnvArgs
import os
from typing import overload


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


# TODO: 需要针对buffer模式优化
# TODO: 加入同步机制
class BasePool(abc.ABC):
    def __init__(self, env_args: EnvArgs):
        self.mean = None
        self.std = None
        self.observation_dims = env_args.observation_dims
        self.action_dims = env_args.action_dims
        self.total_steps = env_args.total_steps

        self.env_args = env_args

        self.observation_buffer = None
        self.next_observation_buffer = None
        self.action_buffer = None
        self.reward_buffer = None  # store step reward

        self.prob_buffer = None

        self.advantage_buffer = np.zeros((self.total_steps, 1), dtype=np.float32)
        self.target_buffer = np.zeros((self.total_steps, 1), dtype=np.float32)
        self.mask_buffer = np.zeros((self.total_steps, 1), dtype=np.int32)

        self.average_rewards_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)
        self.max_rewards_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)
        self.min_rewards_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)

        self.average_traj_len_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)
        self.min_traj_len_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)
        self.max_traj_len_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)

        self.traj_num_buffer = np.zeros((env_args.worker_num, 1), dtype=np.float32)
        #
        if self.env_args.train_rnn_r2d2:
            if env_args.model_args.using_lstm:
                self.hidden_buffer = np.zeros((env_args.total_steps, env_args.model_args.rnn_units * 2),
                                              dtype=np.float32)
            else:
                self.hidden_buffer = np.zeros((env_args.total_steps, env_args.model_args.rnn_units),
                                              dtype=np.float32)

        # counter
        self.pointer = 0  # indicate how much data you  store
        self.last_pointer = 0
        self.summary_pointer = 0

        self.traj_info_is_ok = False

    def store(self, observation, action, reward, mask, next_observation, prob=None):
        self._store(observation, action, reward, mask, next_observation, prob)

    def get_data_slice(self, pointer, last_pointer):
        path = slice(last_pointer, pointer)
        observation = self.observation_buffer[path]
        reward = self.reward_buffer[path]
        return observation, reward

    # @overload
    def _store(self, observation, action, reward, mask, next_observation, prob):
        """
        store your data
        :return:
        you can rewrite this part.
        """
        index = self.pointer
        self.observation_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.mask_buffer[index] = mask
        if prob is not None:
            self.prob_buffer[index] = prob
        if next_observation is not None:
            self.next_observation_buffer[index] = next_observation
        # print(self.prob_buffer[self.pointer], prob.numpy())

        self.pointer += 1

    def store_rnn(self, observation, action, reward, mask, next_observation, prob, hidden):
        index = self.pointer
        self.observation_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.mask_buffer[index] = mask
        if prob is not None:
            self.prob_buffer[index] = prob
        if next_observation is not None:
            self.next_observation_buffer[index] = next_observation

        if hidden is not None:
            self.hidden_buffer[index] = hidden

    def rest_pointer(self):
        if self.env_args.buffer_size is None:
            self.pointer = 0

        self.last_pointer = 0
        self.traj_info_is_ok = False

    def summary_trajs(self, average_reward, max_reward, min_reward, average_traj_len, max_traj_len, min_traj_len,
                      traj_num):
        self.average_rewards_buffer[self.summary_pointer] = average_reward
        self.max_rewards_buffer[self.summary_pointer] = max_reward
        self.min_rewards_buffer[self.summary_pointer] = min_reward
        self.average_traj_len_buffer[self.summary_pointer] = average_traj_len
        self.max_traj_len_buffer[self.summary_pointer] = max_traj_len
        self.min_traj_len_buffer[self.summary_pointer] = min_traj_len
        self.traj_num_buffer[self.summary_pointer] = traj_num

        self.traj_info_is_ok = True

    @property
    def get_average_reward(self):
        total_reward = self.average_rewards_buffer * self.traj_num_buffer
        average_reward = np.sum(total_reward) / np.sum(self.traj_num_buffer)
        return average_reward

    @property
    def get_average_traj_len(self):
        total_trajs = self.average_traj_len_buffer * self.traj_num_buffer
        average_trajs = np.sum(total_trajs) / np.sum(self.traj_num_buffer)
        return average_trajs

    @property
    def get_max_traj_len(self):
        return np.max(self.max_traj_len_buffer)

    @property
    def get_min_traj_len(self):
        return np.max(self.min_traj_len_buffer)

    @property
    def get_max_reward(self):
        return np.max(self.max_rewards_buffer)

    @property
    def get_min_reward(self):
        return np.min(self.min_rewards_buffer)

    @property
    def get_total_trajs(self):
        return np.sum(self.traj_num_buffer)

    @property
    def get_state_std(self):
        if self.std is not None:
            std = self.std
        else:
            std = np.std(self.observation_buffer)

        return std

    @property
    def get_state_mean(self):
        if self.mean is not None:
            mean = self.mean
        else:
            mean = np.mean(self.observation_buffer)

        return mean

    @staticmethod
    def convert_to_tensor(data):
        # out = deepcopy(data)
        return tf.convert_to_tensor(data, dtype=tf.float32)

    def pool_info(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("observation_buffer's shape:{}".format(self.observation_buffer.shape))
        logging.info("action_buffer's shape:{}".format(self.action_buffer.shape))
        logging.basicConfig(level=logging.WARNING)

    def traj_info(self):
        print("Trajectory information:")
        print("Trajectory numbers:{}".format(self.get_total_trajs))
        print("Max reward:{}".format(self.get_max_reward))
        print("Min reward:{}".format(self.get_min_reward))
        print("Average reward:{}".format(self.get_average_reward))
        mean_len = self.get_average_traj_len
        max_len = self.get_max_traj_len
        min_len = self.get_min_traj_len

        if mean_len == max_len:
            pass
        else:
            print("Max trajs len:{}".format(max_len))
            print("Min trajs len:{}".format(min_len))
            print("Average trajs len:{}".format(mean_len))

    def normalize_state(self, bias=1e-3):
        mean = np.mean(self.observation_buffer)
        std = np.std(self.observation_buffer) + bias

        state_buffer = (self.observation_buffer - mean) / std

        next_state_buffer = (self.next_observation_buffer - mean) / std

        self.mean = mean
        self.std = std

        return state_buffer, next_state_buffer, mean, std

    def normalize_reward(self):
        mean = np.mean(self.reward_buffer)
        std = np.std(self.reward_buffer)

        rewards = (self.reward_buffer - mean) / std

        return rewards

    def save_data(self, space):
        mkdir(space)
        np.save(space + '/observation_buffer.npy', self.observation_buffer)
        np.save(space + '/next_observation_buffer.npy', self.next_observation_buffer)
        np.save(space + '/action_buffer.npy', self.action_buffer)
        np.save(space + '/reward_buffer.npy', self.reward_buffer)
        np.save(space + '/prob_buffer.npy', self.prob_buffer)

    def load_data(self, space):
        self.observation_buffer = np.load(space + '/observation_buffer.npy')
        self.next_observation_buffer = np.load(space + '/next_observation_buffer.npy')
        self.action_buffer = np.load(space + '/action_buffer.npy')
        self.reward_buffer = np.load(space + '/reward_buffer.npy')
        self.prob_buffer = np.load(space + '/prob_buffer.npy')
