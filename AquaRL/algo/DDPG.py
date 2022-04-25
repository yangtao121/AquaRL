from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import DDPGParameter
from AquaRL.pool.LocalPool import LocalPool
import numpy as np


# TODO: 需要实现对应的policy
class DDPG(BaseAlgo):
    def __init__(self, hyper_parameters: DDPGParameter, data_pool: LocalPool, actor, critic, work_space):
        """
        DDPG由于一些技术问题，model的不支持专家模式的创建但支持其他两种方式的创建。
        :param hyper_parameters:
        :param data_pool:
        :param actor:
        :param critic:
        :param work_space:
        """
        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=work_space)

        self.actor = actor
        self.critic = critic

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)
        self.train_times = 0

    def _optimize(self):
        sample_range = min(self.data_pool.pointer, self.hyper_parameters.buffer_size)
        sample_index = np.random.choice(sample_range, self.hyper_parameters.batch_size)

        state_batch = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer[sample_index])
        action_batch = self.data_pool.convert_to_tensor(self.data_pool.action_buffer[sample_index])
        reward_batch = self.data_pool.convert_to_tensor(self.data_pool.reward_buffer[sample_index])
        next_state_batch = self.data_pool.convert_to_tensor(self.data_pool.next_observation_buffer[sample_index])
        mask_batch = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer[sample_index])

        q_loss = self.train_critic(state_batch, next_state_batch, action_batch, reward_batch, mask_batch)

        actor_loss = self.train_actor(state_batch)

        # print("q loss:{}".format(q_loss))
        # print('actor loss:{}'.format(actor_loss))

        with self.max_summary_writer.as_default():
            tf.summary.scalar("DDPG/q_loss", q_loss, self.train_times)
            tf.summary.scalar("DDPG/actor_loss", actor_loss, self.train_times)

        self.actor.soft_update(self.hyper_parameters.soft_update_ratio)
        self.critic.soft_update(self.hyper_parameters.soft_update_ratio)

        return q_loss, actor_loss

    def optimize(self):
        if self.data_pool.traj_info_is_ok:
            self.epoch += 1
            print("_______________epoch:{}____________________".format(self.epoch))
            with self.average_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_average_reward, step=self.epoch)
            with self.max_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_max_reward, step=self.epoch)
            with self.min_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_min_reward, step=self.epoch)

            mean_len = self.data_pool.get_average_traj_len
            max_len = self.data_pool.get_max_traj_len
            min_len = self.data_pool.get_min_traj_len

            if max_len == max_len:
                pass
            else:
                with self.average_summary_writer.as_default():
                    tf.summary.scalar("Traj_Info/Len", mean_len, step=self.epoch)
                with self.max_summary_writer.as_default():
                    tf.summary.scalar("Traj_Info/Len", max_len, step=self.epoch)
                with self.min_summary_writer.as_default():
                    tf.summary.scalar("Traj_Info/Len", min_len, step=self.epoch)
            self.data_pool.traj_info()

        q_loss, actor_loss = self._optimize()

        return q_loss, actor_loss

    @tf.function
    def train_critic(self, state_batch, next_state_batch, action_batch, reward_batch, mask_batch):
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_action(next_state_batch)
            y = reward_batch + self.hyper_parameters.gamma * mask_batch * self.critic.target_value(next_state_batch,
                                                                                                   target_actions)
            critic_value = self.critic.online_value(state_batch, action_batch)
            critic_loss = tf.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic.get_variable())
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.get_variable()))

        return critic_loss

    @tf.function
    def train_actor(self, state_batch):
        with tf.GradientTape() as tape:
            actions = self.actor.online_action(state_batch)
            critic_value = self.critic.online_value(state_batch, actions)
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.get_variable())
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.get_variable()))
        return actor_loss

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            actor_learning_rate = 'actor learning rate:{}'.format(self.hyper_parameters.policy_learning_rate)
            critic_learning_rate = 'critic learning rate:{}'.format(self.hyper_parameters.critic_learning_rate)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)
            gamma = 'gamma:{}'.format(self.hyper_parameters.gamma)
            soft_update_ratio = 'soft update ratio:{}'.format(self.hyper_parameters.soft_update_ratio)

            tf.summary.text('DDPG_parameter', actor_learning_rate, step=self.epoch)
            tf.summary.text('DDPG_parameter', critic_learning_rate, step=self.epoch)
            tf.summary.text('DDPG_parameter', batch_size, step=self.epoch)
            tf.summary.text('DDPG_parameter', gamma, step=self.epoch)
            tf.summary.text('DDPG_parameter', soft_update_ratio, step=self.epoch)
