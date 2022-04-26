from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import TD3Parameter
from AquaRL.pool.LocalPool import LocalPool
import numpy as np


# TD3的offline版本
class TD3_BC(BaseAlgo):

    def __init__(self, hyper_parameters: TD3Parameter, data_pool: LocalPool, actor_policy, q_policy1, q_policy2, noise,
                 work_space):
        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=work_space)

        self.actor_policy = actor_policy
        self.q_policy1 = q_policy1
        self.q_policy2 = q_policy2
        self.noise = noise

        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.train_times = 0
        self.actor_loss = None
        self.mse_loss = None

        if self.hyper_parameters.stationary_buffer:
            if self.hyper_parameters.state_normalize:
                self.observation_buffer, self.next_observation_buffer, _, _ = self.data_pool.normalize_state(
                    self.hyper_parameters.bias)
            else:
                self.observation_buffer = self.data_pool.observation_buffer
                self.next_observation_buffer = self.data_pool.next_observation_buffer

            # self.observation_buffer = self.data_pool.convert_to_tensor(self.observation_buffer)
            # self.next_observation_buffer = self.data_pool.convert_to_tensor(self.next_observation_buffer)

            if self.hyper_parameters.reward_normalize:
                self.reward_buffer = self.data_pool.normalize_reward()
            else:
                self.reward_buffer = self.data_pool.reward_buffer

            # self.reward_buffer = self.data_pool.convert_to_tensor(self.reward_buffer)

    def _optimize(self):
        sample_range = min(self.data_pool.pointer, self.hyper_parameters.buffer_size)
        sample_index = np.random.choice(sample_range, self.hyper_parameters.batch_size)
        self.train_times += 1
        if self.hyper_parameters.stationary_buffer:
            reward_batch = self.data_pool.convert_to_tensor(self.reward_buffer[sample_index])
            next_state_batch = self.data_pool.convert_to_tensor(self.next_observation_buffer[sample_index])
            state_batch = self.data_pool.convert_to_tensor(self.observation_buffer[sample_index])
        else:
            if self.hyper_parameters.state_normalize:
                observation_buffer, next_observation_buffer, _, _ = self.data_pool.normalize_state(
                    self.hyper_parameters.bias)
            else:
                observation_buffer = self.data_pool.observation_buffer
                next_observation_buffer = self.data_pool.next_observation_buffer
            if self.hyper_parameters.reward_normalize:
                reward_buffer = self.data_pool.normalize_reward()
            else:
                reward_buffer = self.data_pool.reward_buffer
            state_batch = self.data_pool.convert_to_tensor(observation_buffer[sample_index])

            reward_batch = self.data_pool.convert_to_tensor(reward_buffer[sample_index])
            next_state_batch = self.data_pool.convert_to_tensor(next_observation_buffer[sample_index])

        mask_batch = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer[sample_index])
        action_batch = self.data_pool.convert_to_tensor(self.data_pool.action_buffer[sample_index])

        noise, _ = tf.clip_by_value(
            self.noise(self.hyper_parameters.batch_size),
            -self.hyper_parameters.eval_noise_clip, self.hyper_parameters.eval_noise_clip)

        target_action = self.actor_policy.target_action(state_batch) + noise

        y = reward_batch + self.hyper_parameters.gamma * mask_batch * tf.math.minimum(
            self.q_policy1.target_value(next_state_batch, target_action),
            self.q_policy2.target_value(next_state_batch, target_action))

        q1_loss, q2_loss = self.train_critic(state_batch, action_batch, y)

        if self.train_times % self.hyper_parameters.policy_update_interval:
            self.actor_loss, self.mse_loss = self.train_actor(state_batch, action_batch)
            self.actor_policy.soft_update(self.hyper_parameters.soft_update_ratio)
            self.q_policy1.soft_update(self.hyper_parameters.soft_update_ratio)
            self.q_policy2.soft_update(self.hyper_parameters.soft_update_ratio)

        return q1_loss, q2_loss, self.actor_loss, self.mse_loss

    def optimize(self):
        # if self.data_pool.traj_info_is_ok:
        #     self.epoch += 1
        #     print("_______________epoch:{}____________________".format(self.epoch))
        #     with self.average_summary_writer.as_default():
        #         tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_average_reward, step=self.epoch)
        #     with self.max_summary_writer.as_default():
        #         tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_max_reward, step=self.epoch)
        #     with self.min_summary_writer.as_default():
        #         tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_min_reward, step=self.epoch)
        #
        #     mean_len = self.data_pool.get_average_traj_len
        #     max_len = self.data_pool.get_max_traj_len
        #     min_len = self.data_pool.get_min_traj_len
        #
        #     if max_len == max_len:
        #         pass
        #     else:
        #         with self.average_summary_writer.as_default():
        #             tf.summary.scalar("Traj_Info/Len", mean_len, step=self.epoch)
        #         with self.max_summary_writer.as_default():
        #             tf.summary.scalar("Traj_Info/Len", max_len, step=self.epoch)
        #         with self.min_summary_writer.as_default():
        #             tf.summary.scalar("Traj_Info/Len", min_len, step=self.epoch)
        # self.data_pool.traj_info()

        q1_loss, q2_loss, actor_loss, mse_loss = self._optimize()

        return q1_loss, q2_loss, actor_loss, mse_loss

    @tf.function
    def train_actor(self, state_batch, action_batch):
        Q = self.q_policy1.online_value(state_batch, action_batch)
        lmbda = self.hyper_parameters.alpha / (tf.reduce_mean(tf.abs(Q)))
        # TODO: 这里可能有提升得地方
        with tf.GradientTape() as tape:
            actions = self.actor_policy.online_action(state_batch)
            critic_value = self.q_policy1.online_value(state_batch, actions)
            actor_loss = -tf.reduce_mean(lmbda * critic_value - tf.square(actions - action_batch))
        actor_grad = tape.gradient(actor_loss, self.actor_policy.get_variable())
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_policy.get_variable()))

        mse_loss = tf.reduce_mean(tf.square(actions - action_batch))

        return actor_loss, mse_loss

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

            tf.summary.text('TD3_BC_parameter', actor_learning_rate, step=self.epoch)
            tf.summary.text('TD3_BC_parameter', critic_learning_rate, step=self.epoch)
            tf.summary.text('TD3_BC_parameter', batch_size, step=self.epoch)
            tf.summary.text('TD3_BC_parameter', gamma, step=self.epoch)
            tf.summary.text('TD3_BC_parameter', soft_update_ratio, step=self.epoch)
