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
            self.value_buffer = np.zeros((self.total_steps, 1), dtype=np.float32)
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

    def store_rnn(self, observation, action, reward, mask, next_observation, prob, value, hidden):
        index = self.pointer
        self.observation_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.mask_buffer[index] = mask
        if prob is not None:
            self.prob_buffer[index] = prob
        if next_observation is not None:
            self.next_observation_buffer[index] = next_observation

        self.hidden_buffer[index] = hidden
        self.value_buffer[index] = value

        self.pointer += 1

    def r2d2_data_process(self, input1=None, input2=None):
        """
        [batchsize,timestep,]
        :return:
        """
        indexes = np.where(self.mask_buffer == 0)[0] + 1
        # print(indexes)

        start_index = 0

        move_step = self.env_args.model_args.traj_len - self.env_args.model_args.over_lap_size

        trajs_obs = []
        trajs_act = []
        trajs_reward = []
        trajs_next_obs = []
        trajs_prob = []
        trajs_hidden = []
        trajs_value = []

        burn_ins_obs = []

        trajs_input1 = None
        trajs_input2 = None

        if input1 is not None:
            trajs_input1 = []
        if input2 is not None:
            trajs_input2 = []

        for end_index in indexes:
            data = self.slice_data(start_index, end_index)

            observations = data['observations']
            actions = data['actions']
            rewards = data['rewards']
            next_observations = data['next_observations']
            probs = data['probs']
            hiddens = data['hiddens']
            values = data['values']
            if input1 is not None:
                inp1 = input1[start_index:end_index]
            if input2 is not None:
                inp2 = input2[start_index: end_index]

            start_ind = 0
            end_ind = start_ind + self.env_args.model_args.traj_len

            while end_ind <= end_index:

                traj_obs = observations[start_ind:end_ind]
                traj_act = actions[start_ind: end_ind]
                traj_reward = rewards[start_ind: end_ind]
                traj_next_obs = next_observations[start_ind: end_ind]
                traj_prob = probs[start_ind: end_ind]
                traj_value = values[start_ind: end_ind]

                if input1 is not None:
                    traj_inp1 = inp1[start_ind:end_ind]
                if input2 is not None:
                    traj_inp2 = inp2[start_ind: end_ind]

                burn_in_obs, traj_obs = np.split(traj_obs, (self.env_args.model_args.burn_in,))
                _, traj_act = np.split(traj_act, (self.env_args.model_args.burn_in,))
                _, traj_reward = np.split(traj_reward, (self.env_args.model_args.burn_in,))
                _, traj_next_obs = np.split(traj_next_obs, (self.env_args.model_args.burn_in,))
                _, traj_prob = np.split(traj_prob, (self.env_args.model_args.burn_in,))
                _, traj_value = np.split(traj_value, (self.env_args.model_args.burn_in,))

                if input1 is not None:
                    _, traj_inp1 = np.split(traj_inp1, (self.env_args.model_args.burn_in,))
                if input2 is not None:
                    _, traj_inp2 = np.split(traj_inp2, (self.env_args.model_args.burn_in,))

                if start_ind == 0:
                    traj_hidden = np.zeros((1, self.env_args.model_args.rnn_units * 2))
                else:
                    traj_hidden = hiddens[start_ind - 1]

                trajs_obs.append(np.expand_dims(traj_obs, axis=0))
                burn_ins_obs.append(np.expand_dims(burn_in_obs, axis=0))
                trajs_act.append(np.expand_dims(traj_act, axis=0))
                trajs_reward.append(np.expand_dims(traj_reward, axis=0))
                trajs_next_obs.append(np.expand_dims(traj_next_obs, axis=0))
                trajs_prob.append(np.expand_dims(traj_prob, axis=0))
                trajs_value.append(np.expand_dims(traj_value, axis=0))
                trajs_hidden.append(traj_hidden)

                if input1 is not None:
                    trajs_input1.append(np.expand_dims(traj_inp1, axis=0))
                if input2 is not None:
                    trajs_input2.append(np.expand_dims(traj_inp2, axis=0))

                start_ind = start_ind + move_step
                end_ind = start_ind + self.env_args.model_args.traj_len

        trajs_obs = np.vstack(trajs_obs)
        trajs_act = np.vstack(trajs_act)
        trajs_reward = np.vstack(trajs_reward)
        trajs_next_obs = np.vstack(trajs_next_obs)
        trajs_prob = np.vstack(trajs_prob)
        trajs_hidden = np.vstack(trajs_hidden)

        burn_ins_obs = np.vstack(burn_ins_obs)

        trajs_value = np.vstack(trajs_value)

        if input1 is not None:
            trajs_input1 = np.vstack(trajs_input1)
        if input2 is not None:
            trajs_input2 = np.vstack(trajs_input2)

        return trajs_obs, burn_ins_obs, trajs_act, trajs_reward, trajs_next_obs, trajs_prob, trajs_value, trajs_hidden, trajs_input1, trajs_input2

    def slice_data(self, start_index, end_index):
        observations = self.observation_buffer[start_index:end_index]
        actions = self.action_buffer[start_index: end_index]
        rewards = self.reward_buffer[start_index: end_index]
        probs = self.prob_buffer[start_index: end_index]
        next_observations = self.next_observation_buffer[start_index: end_index]

        data = {'observations': observations, 'actions': actions, 'rewards': rewards,
                'next_observations': next_observations, 'probs': probs}

        if self.env_args.train_rnn_r2d2:
            hiddens = self.hidden_buffer[start_index: end_index]
            values = self.value_buffer[start_index: end_index]
            data['hiddens'] = hiddens
            data['values'] = values

        return data

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
