from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
import tensorflow_probability as tfp
from copy import deepcopy


# TODO: clear data shape
class PPO(BaseAlgo):
    def __init__(self, hyper_parameters, data_pool, actor,
                 critic, works_pace=None, discriminator=None):
        super().__init__(hyper_parameters, data_pool, works_pace)
        self.actor = actor
        self.critic = critic

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)
        self.discriminator = discriminator

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        # tf_next_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.next_observation_buffer)

        tf_values_buffer = self.critic.get_value(tf_observation_buffer)
        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        tf_old_probs = self.data_pool.convert_to_tensor(self.data_pool.prob_buffer)
        # tf_next_values_buffer = self.critic.get_value(tf_next_observation_buffer)

        # tf_mask_buffer = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer)
        if self.discriminator is None:
            tf.reward_buffer = self.data_pool.convert_to_tensor(self.data_pool.reward_buffer)
        else:
            tf.reward_buffer = self.discriminator.get_rewards_buffer(tf_observation_buffer, tf_action_buffer)

        gae, target = self.cal_gae_target(self.data_pool.reward_buffer, tf_values_buffer.numpy(),
                                          self.data_pool.mask_buffer)
        tf_gae = self.data_pool.convert_to_tensor(gae)
        tf_target = self.data_pool.convert_to_tensor(target)

        max_steps = self.data_pool.max_steps
        # print(max_steps)

        for _ in tf.range(0, self.hyper_parameters.update_steps):
            start_pointer = 0
            end_pointer = self.hyper_parameters.batch_size - 1
            while end_pointer <= max_steps - 1:
                state = tf_observation_buffer[start_pointer: end_pointer]
                action = tf_action_buffer[start_pointer: end_pointer]
                gae = tf_gae[start_pointer: end_pointer]
                target = tf_target[start_pointer: end_pointer]
                old_prob = tf_old_probs[start_pointer: end_pointer]

                self.train_actor(state, action, gae, old_prob)
                self.train_critic(state, target)
                # print(acc_loss)
                # print(ok)
                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

    @tf.function
    def train_critic(self, observation, target):
        """
        inputs are tf.tensor.
        :param observation:
        :param target:
        :return: tensor
        """
        if self.hyper_parameters.clip_critic_value:
            with tf.GradientTape() as tape:
                v = self.critic(observation)
                surrogate1 = tf.square(v[1:] - target[1:])
                surrogate2 = tf.square(
                    tf.clip_by_value(v[1:], v[:-1] - self.hyper_parameters.clip_ratio,
                                     v[:-1] + self.hyper_parameters.clip_ratio) - target[1:])
                critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            with tf.GradientTape() as tape:
                v = self.critic(observation)
                critic_loss = tf.reduce_mean(tf.square(target - v))

        grad = tape.gradient(critic_loss, self.critic.get_variable())
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.get_variable()))

        return critic_loss

    @tf.function
    def train_actor(self, state, action, advantage, old_prob):
        # TODO:Add new function entropy loss.
        """
        inputs are tf.tensor
        :param state:
        :param action:
        :param advantage:
        :param old_prob:
        :return:
        """
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(state)
            pi = tfp.distributions.Normal(mu, sigma)

            # print(action.shape)
            # print(old_prob.shape)
            ratio = tf.clip_by_value(pi.prob(action), 1e-6, 1) / old_prob
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                     1 + self.hyper_parameters.clip_ratio) * advantage
                )
            )
        actor_grad = tape.gradient(actor_loss, self.actor.get_variable())
        # print(actor_grad)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.get_variable()))

        return actor_loss
