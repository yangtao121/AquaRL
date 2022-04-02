import abc
import numpy as np
import tensorflow as tf
import datetime


# TODO: clear data shape
class BaseAlgo(abc.ABC):
    def __init__(self, hyper_parameters, data_pool):
        self.hyper_parameters = hyper_parameters
        self.data_pool = data_pool  # memery share

        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.min_summary_writer = tf.summary.create_file_writer(log_dir + "/min")
        self.max_summary_writer = tf.summary.create_file_writer(log_dir + "/max")
        self.main_summary_writer = tf.summary.create_file_writer(log_dir + "/average")

        self.epoch = 0

    def cal_discount_reward(self, rewards, mask):
        discount_rewards = []
        mask_ = mask[::-1]
        value_ = 0
        for i, reward in enumerate(rewards[::-1]):
            value_ = reward + self.hyper_parameters.gamma * value_ * mask_[i]
            discount_rewards.append(value_)

        discount_rewards.reverse()
        discount_rewards = np.hstack(discount_rewards)

        return discount_rewards

    def cal_gae_target(self, rewards, values, mask):
        gae = np.zeros_like(rewards)
        n_steps_target = np.zeros_like(rewards)
        cumulate_gae = 0
        next_val = 0

        for i in reversed(range(0, len(rewards))):
            delta = rewards[i] + self.hyper_parameters.gamma * next_val - values[i]
            cumulate_gae = self.hyper_parameters.gamma * self.hyper_parameters.lambada * cumulate_gae * mask[i] + delta
            gae[i] = cumulate_gae
            next_val = values[i]
            n_steps_target[i] = gae[i] + values[i]

        return gae, n_steps_target

    def optimize(self):

        with self.main_summary_writer.as_default():
            tf.summary.scalar("Reward", self.data_pool.get_average_reward, step=self.epoch)
        with self.max_summary_writer.as_default():
            tf.summary.scalar("Reward", self.data_pool.get_max_reward, step=self.epoch)
        with self.min_summary_writer.as_default():
            tf.summary.scalar("Reward", self.data_pool.get_min_reward, step=self.epoch)

        print("_______________epoch:{}____________________".format(self.epoch))
        print("Training:")
        print("Average reward:{}".format(self.data_pool.get_average_reward))
        print("Min reward:{}".format(self.data_pool.get_min_reward))
        print("Max reward:{}".format(self.data_pool.get_max_reward))

        self._optimize()

        self.epoch += 1

    @abc.abstractmethod
    def _optimize(self):
        """
        update model, when you rewrite this part you can add tf.function
        :return:
        """
