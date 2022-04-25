import tensorflow_probability as tfp
import tensorflow as tf


class GaussianNoise:
    def __init__(self, shape, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        mu = tf.zeros(shape=(shape,), dtype=tf.float32)
        sigma = tf.ones(shape=(shape,), dtype=tf.float32)

        self.dist = tfp.distributions.Normal(mu, sigma)

    @tf.function
    def out_and_prob(self, size=None):
        if size is None:
            out = self.dist.sample()
        else:
            out = self.dist.sample(size)

        prob = self.dist.prob(out)

        prob = tf.clip_by_value(prob, 1e-6, 1)
        # prob = tf.squeeze(prob)

        return out, prob

    def __call__(self, size=None):
        noise, prob = self.out_and_prob(size)
        out = noise * self.sigma + self.mu
        if size is None:
            out = tf.squeeze(out)
            prob = tf.squeeze(prob)
        return out, prob


if __name__ == "__main__":
    import numpy as np

    noise = GaussianNoise(3, 0, 1)

    a = np.zeros(4)

    out, prob = noise.out_and_prob(a.shape[0])

    print(out)
    # import numpy as np
    #
    # a = np.array([0, 3, 2])
    # b = np.array([1, 2, 3])
    #
    # c = tf.math.minimum(a,b)
    #
    # print(c.shape[0])
