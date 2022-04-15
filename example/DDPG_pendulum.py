import gym
from AquaRL.args import DDPGParameter, EnvArgs
from AquaRL.algo.DDPG import DDPG

from AquaRL.worker.Worker import Worker
from AquaRL.policy.DeterminedPolicy import DeterminedPolicy
from AquaRL.policy.CriticPolicy import StateActionCriticPolicy
from AquaRL.neural import mlp, state_action_mlp

from AquaRL.pool.LocalPool import LocalPool
import tensorflow as tf
from AquaRL.noise.OUNoise import OrnsteinUhlenbeckActionNoise
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    epochs=100,
    buffer_size=50000,
    step_training=True,
    worker_num=1
)

hyper_parameter = DDPGParameter(
    buffer_size=env_args.buffer_size
)

actor = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor',
    output_activation='tanh'
)

q_value_net = state_action_mlp(
    state_dims=env_args.observation_dims,
    action_dims=env_args.action_dims,
    hidden_size=(8, 64, 64)
)

noise = OrnsteinUhlenbeckActionNoise(
    mu=np.zeros(env_args.action_dims),
    sigma=np.ones(env_args.action_dims)
)

actor_policy = DeterminedPolicy(model=actor, noise=noise, policy_name='actor')
critic_policy = StateActionCriticPolicy(model=q_value_net, policy_name='critic')

data_pool = LocalPool(env_args)
ddpg = DDPG(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor=actor_policy,
    critic=critic_policy,
    work_space='ddpg'
)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=actor_policy, action_fun=action_fun,
                is_distribution=False)

done = False

for i in range(env_args.epochs):
    noise.reset()
    data_pool.rest_pointer()
    done = False
    while not done:
        done = worker.step()
        ddpg.optimize()
