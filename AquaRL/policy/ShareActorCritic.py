import tensorflow as tf
from AquaRL.policy.BasePolicy import BasePolicy
import tensorflow_probability as tfp
import numpy as np


class GaussianActorCriticPolicy(BasePolicy):
    def __init__(self, out_shape, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)
        self.log_std = tf.Variable(tf.zeros((out_shape,)) - 0.5, trainable=True)
        self.mu = tf.zeros(shape=(out_shape,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(out_shape,), dtype=tf.float32)

        self.noise_dist = tfp.distributions.Normal(self.mu, self.sigma)

    def get_actor_critic(self, obs, hidden):
        mu, value = self.Model(obs, initial_state=hidden)
        sigma = tf.exp(self.log_std)

        return mu, value, sigma

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def set_std(self, std):
        self.log_std = tf.Variable(std, dtype=tf.float32)

    def get_std(self):
        return self.log_std.numpy()

    @tf.function
    def get_mu(self, obs):
        return self.Model.actor(obs)

    @tf.function
    def noise_and_prob(self):
        noise = self.noise_dist.sample()
        prob = self.noise_dist.prob(noise)

        prob = tf.clip_by_value(prob, 1e-6, 1)
        prob = tf.squeeze(prob)

        return noise, prob

    def get_action(self, obs):
        """
        obs is tf.tensor
        :param obs:
        :return:
        """
        obs = tf.cast(obs, tf.float32)
        a_adv = self.get_mu(obs)
        a_adv = tf.squeeze(a_adv, axis=0)
        a_std = tf.exp(self.log_std)

        noise, prob = self.noise_and_prob()

        action = a_adv + noise * a_std

        return action, prob

    def action(self, obs):
        return self.get_mu(obs)

    def actor(self, obs):
        mu = self.get_mu(obs)
        sigma = tf.exp(self.log_std)

        return mu, sigma

    @tf.function
    def critic(self, obs):
        return self.Model.critic(obs)

    @tf.function
    def get_value(self, obs):
        v = self.Model.critic(obs)
        v = tf.squeeze(v)
        return v


class LSTMGaussianActorCritic(BasePolicy):
    def __init__(self, out_shape, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)
        self.log_std = tf.Variable(tf.zeros((out_shape,)) - 0.5, trainable=True)
        self.mu = tf.zeros(shape=(out_shape,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(out_shape,), dtype=tf.float32)

        self.noise_dist = tfp.distributions.Normal(self.mu, self.sigma)

    @tf.function
    def noise_and_prob(self):
        noise = self.noise_dist.sample()
        prob = self.noise_dist.prob(noise)

        prob = tf.clip_by_value(prob, 1e-6, 1)
        prob = tf.squeeze(prob)

        return noise, prob

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def get_action(self, obs, done):
        # obs = tf.expand_dims(obs, axis=0)
        obs = tf.reshape(obs, [1, 1, -1])
        # obs = tf.expand_dims(obs, axis=0)
        mu, value, hidden_state = self.Model(obs, done=done)
        a_std = tf.exp(self.log_std)
        mu = tf.squeeze(mu)
        value = tf.squeeze(value)

        hidden_state = np.hstack(hidden_state)

        noise, prob = self.noise_and_prob()

        action = mu + noise * a_std

        return action, prob, hidden_state, value

    def get_actor_critic(self, obs, training=True):
        mu, value, _ = self.Model(obs, training=training)
        sigma = tf.exp(self.log_std)

        return mu, value, sigma

    def set_std(self, std):
        self.log_std = tf.Variable(std, dtype=tf.float32)

    def get_std(self):
        return self.log_std.numpy()
