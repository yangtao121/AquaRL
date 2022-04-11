import tensorflow as tf
import tensorflow_probability as tfp
from AquaRL.policy.BasePolicy import BasePolicy


# TODO:need to be checked
class GaussianPolicy(BasePolicy):
    def __init__(self, out_shape, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)

        self.log_std = tf.Variable(tf.zeros((out_shape,)) - 0.5, trainable=True)
        self.mu = tf.zeros(shape=(out_shape,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(out_shape,), dtype=tf.float32)

        self.noise_dist = tfp.distributions.Normal(self.mu, self.sigma)

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def set_std(self, std):
        self.log_std = tf.Variable(std, dtype=tf.float32)

    def get_std(self):
        return self.log_std.numpy()

    @tf.function
    def get_mu(self, obs):
        return self.Model(obs)

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

    def __call__(self, obs):
        mu = self.get_mu(obs)
        sigma = tf.exp(self.log_std)

        return mu, sigma


class COAdaptiveStd(BasePolicy):
    """
    网络包含mu，sigma
    """

    def __init__(self, out_shape, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)
        self.mu = tf.zeros(shape=(out_shape,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(out_shape,), dtype=tf.float32)

        self.noise_dist = tfp.distributions.Normal(self.mu, self.sigma)

    def __call__(self, obs):
        [mu, sigma] = self.Model(obs)
        std = tf.exp(sigma)
        return mu, std

    @tf.function
    def noise_and_prob(self):
        noise = self.noise_dist.sample()
        prob = self.noise_dist.prob(noise)

        prob = tf.clip_by_value(prob, 1e-6, 1)
        prob = tf.squeeze(prob)

        return noise, prob

    def get_variable(self):
        return self.Model.trainable_variables

    def get_mu(self, obs):
        mu, sigma = self(obs)
        return mu

    def get_action(self, obs):
        obs = tf.cast(obs, tf.float32)
        mu, std = self(obs)
        noise, prob = self.noise_and_prob()

        action = mu + noise * std

        return action, prob

