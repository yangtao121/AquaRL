import gym
from AquaRL.args import PPOHyperParameters, EnvArgs
from AquaRL.algo.PPO import PPO

from AquaRL.worker.Worker import Worker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.neural import mlp
from AquaRL.pool.LocalPool import LocalPool
import tensorflow as tf
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
    total_steps=4000,
    epochs=100,
    worker_num=1
)

hyper_parameter = PPOHyperParameters(
    batch_size=200
)

actor = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor'
)

value_net = mlp(
    state_dims=env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    name='value'
)

policy = GaussianPolicy(out_shape=1, model=actor, file_name='policy')
critic = CriticPolicy(model=value_net, file_name='critic')

data_pool = LocalPool(env_args=env_args)

ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor=policy,
    critic=critic
)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy, action_fun=action_fun)
max_mark = -1e6
for i in range(env_args.epochs):
    worker.sample()
    ppo.optimize()

# policy.save_model('actor.h5')
