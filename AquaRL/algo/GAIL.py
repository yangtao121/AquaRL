from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
import numpy as np
from AquaRL.args import GAILParameters


# TODO: change work type
class GAIL(BaseAlgo):
    def __init__(self, parameters: GAILParameters, expert_data_pool, data_pool, discriminator, work_space=None):
        super().__init__(hyper_parameters=parameters, data_pool=data_pool, work_space=work_space)

        self.expert_data_pool = expert_data_pool

        self.discriminator = discriminator

        # merge data
        concat_data = np.concatenate((self.expert_data_pool.observation_buffer, self.expert_data_pool.action_buffer),
                                     axis=1)
        self.tf_expert_s_a = self.expert_data_pool.convert_to_tensor(concat_data)

        self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.learning_rate)

        self.tf_expert_states = self.expert_data_pool.convert_to_tensor(self.expert_data_pool.observation_buffer)
        self.tf_expert_actions = self.expert_data_pool.convert_to_tensor(self.expert_data_pool.action_buffer)

        self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.learning_rate)

    def _optimize(self):
        tf_states = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        tf_actions = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)

        for i in range(self.hyper_parameters.update_times):
            self.train_discriminator(tf_states, tf_actions)

        entropy, entropy_agent, entropy_expert = self.cal_entropy(tf_states, tf_actions)

        print("Entropy:{}".format(entropy))

        with self.main_summary_writer.as_default():
            tf.summary.scalar("GAIL/entropy", entropy, self.epoch)
            tf.summary.scalar("GAIL/expert_entropy", entropy_expert, self.epoch)
            tf.summary.scalar("GAIL/agent_entropy", entropy_agent, self.epoch)

    def optimize(self):
        self._optimize()
        self.epoch += 1

    @tf.function
    def cal_entropy(self, state, action):
        expert_reward = self.discriminator(self.tf_expert_states, self.tf_expert_actions)
        reward = self.discriminator(state, action)

        loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(expert_reward, 0.01, 1)))
        loss_agent = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - reward, 0.01, 1)))
        loss = loss_agent + loss_expert
        return loss, loss_agent, loss_expert

    @tf.function
    def train_discriminator(self, state, action):
        with tf.GradientTape() as tape:
            expert_reward = self.discriminator(self.tf_expert_states, self.tf_expert_actions)
            reward = self.discriminator(state, action)

            loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(expert_reward, 0.01, 1)))
            loss_agent = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - reward, 0.01, 1)))
            loss = loss_agent + loss_expert
            loss = -loss

        grad = tape.gradient(loss, self.discriminator.get_variable())
        self.discriminator_optimizer.apply_gradients(zip(grad, self.discriminator.get_variable()))

        return loss

    def write_parameter(self):
        learning_rate = 'learning rate:{}'.format(self.hyper_parameters.learning_rate)
        update_times = 'update times:{}'.format(self.hyper_parameters.update_times)

        with self.main_summary_writer.as_default():
            tf.summary.text('GAIL_parameter', learning_rate, step=self.epoch)
            tf.summary.text('GAIL_parameter', update_times, step=self.epoch)
