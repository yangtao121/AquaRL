from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import TD3Parameter
from AquaRL.pool.LocalPool import LocalPool
import numpy as np


# Twin Delayed Deep Deterministic policy gradient algorithm

class TD3(BaseAlgo):

    def __init__(self, hyper_parameters: TD3Parameter, data_pool: LocalPool, actor_policy, q_policy1, q_policy2, noise,
                 work_space):
        """
        从TD3开始policy风格将逐渐更新。
        :param hyper_parameters:
        :param data_pool:
        :param actor_policy:
        :param work_space:
        """

        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=work_space)

        self.actor_policy = actor_policy
        self.q_policy1 = q_policy1
        self.q_policy2 = q_policy2
        self.noise = noise

        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.train_times = 0

    def _optimize(self):
        self.train_times += 1
        sample_range = min(self.data_pool.pointer, self.hyper_parameters.buffer_size)
        sample_index = np.random.choice(sample_range, self.hyper_parameters.batch_size)

        state_batch = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer[sample_index])
        action_batch = self.data_pool.convert_to_tensor(self.data_pool.action_buffer[sample_index])
        reward_batch = self.data_pool.convert_to_tensor(self.data_pool.reward_buffer[sample_index])
        next_state_batch = self.data_pool.convert_to_tensor(self.data_pool.next_observation_buffer[sample_index])
        mask_batch = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer[sample_index])

        noise, _ = tf.clip_by_value(
            self.noise(self.hyper_parameters.batch_size),
            -self.hyper_parameters.eval_noise_clip, self.hyper_parameters.eval_noise_clip)

        target_action = self.actor_policy.target_action(state_batch) + noise

        y = reward_batch + self.hyper_parameters.gamma * mask_batch * tf.math.minimum(
            self.q_policy1.target_value(next_state_batch, target_action),
            self.q_policy2.target_value(next_state_batch, target_action))

        q1_loss, q2_loss = self.train_critic(state_batch, action_batch, y)
        actor_loss = None
        if self.train_times % self.hyper_parameters.policy_update_interval:
            actor_loss = self.train_actor(state_batch)
            self.actor_policy.soft_update(self.hyper_parameters.soft_update_ratio)
            self.q_policy1.soft_update(self.hyper_parameters.soft_update_ratio)
            self.q_policy2.soft_update(self.hyper_parameters.soft_update_ratio)

        return q1_loss, q2_loss, actor_loss

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

        q1_loss, q2_loss, actor_loss = self._optimize()

        return q1_loss, q2_loss, actor_loss

    @tf.function
    def train_actor(self, state_batch):
        # TODO: 这里可能有提升得地方
        with tf.GradientTape() as tape:
            actions = self.actor_policy.online_action(state_batch)
            critic_value = self.q_policy1.online_value(state_batch, actions)
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_policy.get_variable())
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_policy.get_variable()))

        return actor_loss

    @tf.function
    def train_critic(self, state_batch, action_batch, y):

        with tf.GradientTape() as q1_tape:
            q1_loss = tf.reduce_mean(
                tf.square(
                    y - self.q_policy1.online_value(state_batch, action_batch)
                )
            )

        q1_grad = q1_tape.gradient(q1_loss, self.q_policy1.get_variable())
        self.critic_optimizer.apply_gradients(zip(q1_grad, self.q_policy1.get_variable()))

        with tf.GradientTape() as q2_tape:
            q2_loss = tf.reduce_mean(
                tf.square(
                    y - self.q_policy2.online_value(state_batch, action_batch)
                )
            )

        q2_grad = q2_tape.gradient(q2_loss, self.q_policy2.get_variable())
        self.critic_optimizer.apply_gradients(zip(q2_grad, self.q_policy2.get_variable()))

        return q1_loss, q2_loss

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            actor_learning_rate = 'actor learning rate:{}'.format(self.hyper_parameters.policy_learning_rate)
            critic_learning_rate = 'critic learning rate:{}'.format(self.hyper_parameters.critic_learning_rate)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)
            gamma = 'gamma:{}'.format(self.hyper_parameters.gamma)
            soft_update_ratio = 'soft update ratio:{}'.format(self.hyper_parameters.soft_update_ratio)

            tf.summary.text('TD3_parameter', actor_learning_rate, step=self.epoch)
            tf.summary.text('TD3_parameter', critic_learning_rate, step=self.epoch)
            tf.summary.text('TD3_parameter', batch_size, step=self.epoch)
            tf.summary.text('TD3_parameter', gamma, step=self.epoch)
            tf.summary.text('TD3_parameter', soft_update_ratio, step=self.epoch)
