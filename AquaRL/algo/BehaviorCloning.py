from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import BCParameter, ActorCriticBehaviorCloningParameter
import numpy as np
import tensorflow_probability as tfp

"""
This is a beta version.
输入是observation，输出是action
"""


class BehaviorCloning(BaseAlgo):
    """
    action是label.
    """

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            learning_rate = 'learning rate:{}'.format(self.hyper_parameters.learning_rate)
            update_times = 'update times:{}'.format(self.hyper_parameters.update_times)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)

            tf.summary.text('BC_parameter', learning_rate, step=self.epoch)
            tf.summary.text('BC_parameter', update_times, step=self.epoch)
            tf.summary.text('BC_parameter', batch_size, step=self.epoch)

    def __init__(self, hyper_parameters: BCParameter, data_pool, policy, workspace):
        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=workspace)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.learning_rate)
        self.policy = policy

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        max_steps = self.data_pool.total_steps

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training before:")
        print("MSE loss:{}".format(loss))
        with self.before_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

        for _ in tf.range(0, self.hyper_parameters.update_times):
            start_pointer = 0
            end_pointer = self.hyper_parameters.batch_size - 1

            while end_pointer <= max_steps:
                state = tf_observation_buffer[start_pointer: end_pointer]
                action = tf_action_buffer[start_pointer: end_pointer]

                self.train_policy(state, action)

                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training after:")
        print("MSE loss:{}".format(loss))
        with self.after_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

    # def optimize(self):

    @tf.function
    def train_policy(self, observation, action):
        with tf.GradientTape() as tape:
            prediction = self.policy(observation)
            mse = tf.reduce_mean(tf.square(prediction - action))
        grad = tape.gradient(mse, self.policy.get_variable())
        self.policy_optimizer.apply_gradients(zip(grad, self.policy.get_variable()))

        return mse

    @tf.function
    def cal_loss(self, observation, action):
        prediction = self.policy(observation)
        # print(prediction[0])
        # print(action)
        mse = tf.reduce_mean(tf.square(prediction - action))

        return mse


# TODO: 这里有一些问题后面在调整
class ActorCriticBehaviorCloning(BaseAlgo):
    def __init__(self, hyper_parameter: ActorCriticBehaviorCloningParameter, actor, critic, expert_data_pool,
                 test_data_pool, work_space):
        super().__init__(hyper_parameters=hyper_parameter, data_pool=test_data_pool, work_space=work_space)
        self.actor = actor
        self.critic = critic
        self.expert_data_pool = expert_data_pool

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)

    def _optimize(self):
        sample_range = min(self.expert_data_pool.pointer, self.hyper_parameters.buffer_size - 1)
        sample_index = np.random.choice(sample_range, self.hyper_parameters.batch_size)

        state_batch = self.expert_data_pool.convert_to_tensor(self.expert_data_pool.observation_buffer[sample_index])
        action_batch = self.expert_data_pool.convert_to_tensor(self.expert_data_pool.action_buffer[sample_index])
        reward_batch = self.expert_data_pool.convert_to_tensor(self.expert_data_pool.reward_buffer[sample_index])
        next_state_batch = self.expert_data_pool.convert_to_tensor(
            self.expert_data_pool.next_observation_buffer[sample_index])

        q_loss = self.train_critic(state_batch, next_state_batch, action_batch, reward_batch)

        actor_loss = self.train_actor(state_batch)

        # print("q loss:{}".format(q_loss))
        # print('actor loss:{}'.format(actor_loss))

        with self.max_summary_writer.as_default():
            tf.summary.scalar("ACBC/q_loss", q_loss, self.epoch)
            tf.summary.scalar("ACBC/actor_loss", actor_loss, self.epoch)

        self.actor.soft_update(self.hyper_parameters.soft_update_ratio)
        self.critic.soft_update(self.hyper_parameters.soft_update_ratio)
        return q_loss, actor_loss

    def optimize(self):
        q_loss, actor_loss = self._optimize()
        if self.data_pool.traj_info_is_ok:
            self.epoch += 1
            # print("_______________epoch:{}____________________".format(self.epoch))
            with self.average_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_average_reward, step=self.epoch)
            with self.max_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_max_reward, step=self.epoch)
            with self.min_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_min_reward, step=self.epoch)

            # print("q loss:{}".format(q_loss))
            # print('actor loss:{}'.format(actor_loss))
        return q_loss, actor_loss

    @tf.function
    def train_critic(self, state_batch, next_state_batch, action_batch, reward_batch, ):
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_action(next_state_batch)
            y = reward_batch + self.hyper_parameters.gamma * self.critic.target_value(next_state_batch, target_actions)
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
        return -actor_loss

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            actor_learning_rate = 'actor learning rate:{}'.format(self.hyper_parameters.policy_learning_rate)
            critic_learning_rate = 'critic learning rate:{}'.format(self.hyper_parameters.critic_learning_rate)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)
            gamma = 'gamma:{}'.format(self.hyper_parameters.gamma)
            soft_update_ratio = 'soft update ratio:{}'.format(self.hyper_parameters.soft_update_ratio)

            tf.summary.text('ACBC_parameter', actor_learning_rate, step=self.epoch)
            tf.summary.text('ACBC_parameter', critic_learning_rate, step=self.epoch)
            tf.summary.text('ACBC_parameter', batch_size, step=self.epoch)
            tf.summary.text('ACBC_parameter', gamma, step=self.epoch)
            tf.summary.text('ACBC_parameter', soft_update_ratio, step=self.epoch)


class DistributionBehaviorCloning(BaseAlgo):
    def __init__(self, hyper_parameters: BCParameter, data_pool, policy, workspace):
        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=workspace)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.learning_rate)
        self.policy = policy

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            learning_rate = 'learning rate:{}'.format(self.hyper_parameters.learning_rate)
            update_times = 'update times:{}'.format(self.hyper_parameters.update_times)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)

            tf.summary.text('DBC_parameter', learning_rate, step=self.epoch)
            tf.summary.text('DBC_parameter', update_times, step=self.epoch)
            tf.summary.text('DBC_parameter', batch_size, step=self.epoch)

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        max_steps = self.data_pool.total_steps
        mu, sigma = self.policy(tf_observation_buffer)
        pi = tfp.distributions.Normal(mu, sigma)
        tf_old_prob_buffer = tf.clip_by_value(pi.prob(tf_action_buffer), 1e-6, 1)

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training before:")
        print("MSE loss:{}".format(loss))
        with self.before_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

        for _ in tf.range(0, self.hyper_parameters.update_times):
            start_pointer = 0
            end_pointer = min(self.hyper_parameters.batch_size - 1, max_steps - 1)

            while end_pointer <= max_steps:
                state = tf_observation_buffer[start_pointer: end_pointer]
                action = tf_action_buffer[start_pointer: end_pointer]
                old_prob = tf_old_prob_buffer[start_pointer: end_pointer]

                self.train_policy(state, action, old_prob)

                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training after:")
        print("MSE loss:{}".format(loss))
        with self.after_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

    @tf.function
    def train_policy(self, observation, action, old_prob):
        with tf.GradientTape() as tape:
            mu, sigma = self.policy(observation)
            pi = tfp.distributions.Normal(mu, sigma)

            new_prob = tf.clip_by_value(pi.prob(action), 1e-6, 1)

            # ratio = new_prob / old_prob
            #
            # loss = -tf.reduce_mean(tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
            #                                         1 + self.hyper_parameters.clip_ratio))
            loss = -new_prob

        grad = tape.gradient(loss, self.policy.get_variable())
        self.policy_optimizer.apply_gradients(zip(grad, self.policy.get_variable()))

    @tf.function
    def cal_loss(self, observation, action):
        prediction, _ = self.policy(observation)
        # print(prediction[0])
        # print(action)
        mse = tf.reduce_mean(tf.square(prediction - action))

        return mse

