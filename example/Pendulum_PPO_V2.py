import gym
from AquaRL.args import PPOHyperParameters, ModelArgs, EnvArgs
from AquaRL.algo.PPO import PPO

from AquaRL.worker.Worker import Worker
from AquaRL.policy.ShareActorCritic import GaussianActorCriticPolicy
from AquaRL.pool.LocalPool import LocalPool
import tensorflow as tf
from model import ActorCritic
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

model_args = ModelArgs(
    share_hidden_param=True
)

env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=4000,
    epochs=100,
    worker_num=1,
    model_args=model_args
)

hyper_parameter = PPOHyperParameters(
    batch_size=128,
    model_args=model_args,
    c1=0.02,
    c2=0.005
)

actor_critic = ActorCritic()

# 初始化网络
actor_critic(tf.random.normal((1, 3), dtype=tf.float32))

policy = GaussianActorCriticPolicy(1, model=actor_critic, file_name='actor_critic')

data_pool = LocalPool(env_args=env_args)

ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor_critic=policy,
    works_pace='share_hidden'
)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy, action_fun=action_fun)

for i in range(env_args.epochs):
    worker.sample()
    ppo.optimize()
