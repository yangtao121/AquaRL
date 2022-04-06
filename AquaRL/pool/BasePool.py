import abc
import logging
import numpy as np
import tensorflow as tf


class BasePool(abc.ABC):
    def __init__(self, observation_dims, action_dims, max_steps, epochs):
        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.max_steps = max_steps
        self.epochs = epochs

        # these should be rewritten according your correspond method.

        self.observation_buffer = None
        self.next_observation_buffer = None
        self.action_buffer = None
        self.reward_buffer = None  # store step reward

        self.episode_reward_buffer = None  # store episode reward

        self.advantage_buffer = np.zeros((max_steps, 1), dtype=np.float32)
        self.target_buffer = np.zeros((max_steps, 1), dtype=np.float32)
        self.mask_buffer = np.zeros((max_steps, 1), dtype=np.int32)

        self.prob_buffer = None

        ############################################################
        # counter
        self.pointer = 0  # indicate how much data you  store
        self.last_pointer = 0
        self.episode_pointer = 0

    def store(self, observation, action, reward, mask, next_observation=None, prob=None):
        self._store(observation, action, reward, mask, next_observation, prob)

    def get_data_slice(self, pointer, last_pointer):
        path = slice(last_pointer, pointer)
        observation = self.observation_buffer[path]
        reward = self.reward_buffer[path]
        return observation, reward

    def _store(self, observation, action, reward, mask, next_observation, prob):
        """
        store your data
        :return:
        you can rewrite this part.
        """
        self.observation_buffer[self.pointer] = observation

        if next_observation is not None and next_observation is not None:
            self.next_observation_buffer[self.pointer] = next_observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.mask_buffer[self.pointer] = mask

        if prob is not None and prob is not None:
            self.prob_buffer[self.pointer] = prob
        # print(self.prob_buffer[self.pointer], prob.numpy())

        self.pointer += 1

    def rest_pointer(self):
        self.pointer = 0
        self.last_pointer = 0
        self.episode_pointer = 0

    def summery_episode(self, episode_reward):
        self.episode_reward_buffer[self.episode_pointer] = episode_reward
        self.episode_pointer += 1

    @staticmethod
    def save_data(data, file_name):
        np.savetxt(file_name, data, delimiter=",")

    def save_all_data(self, file_name):
        self.save_data(self.reward_buffer, file_name + 'reward.csv')
        self.save_data(self.observation_buffer, file_name + 'observation.csv')
        if self.prob_buffer is not None:
            self.save_data(self.prob_buffer, file_name + 'prob.csv')
        self.save_data(self.action_buffer, file_name + 'action.csv')
        self.save_data(self.mask_buffer, file_name + 'mask.csv')

    @property
    def get_average_reward(self):
        return np.mean(self.episode_reward_buffer)

    @property
    def get_max_reward(self):
        return np.max(self.episode_reward_buffer)

    @property
    def get_min_reward(self):
        return np.min(self.episode_reward_buffer)

    @property
    def get_std_reward(self):
        return np.std(self.episode_reward_buffer)

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
        print("Numbers:{}".format(self.epochs))
        print("Max reward:{}".format(self.get_min_reward))
        print("Min reward:{}".format(self.get_max_reward))
        print("Average reward:{}".format(self.get_average_reward))
        print("Std reward:{}".format(self.get_std_reward))
