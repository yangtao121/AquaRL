import gym

env = gym.make("Pendulum-v0")

print(env.action_space)
# import numpy as np
# from AquaRL.neural import gaussian_mlp
#
# mlp = gaussian_mlp(3, 1, (64, 64))
#
# a = np.ones(shape=(1, 3))
#
# [mu, sigma] = mlp(a)
# print(mu)
