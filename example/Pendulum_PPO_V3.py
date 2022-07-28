import gym
import numpy as np

from AquaRL.args import PPOHyperParameters, ModelArgs, EnvArgs
from AquaRL.algo.PPO import PPO

from AquaRL.worker.Worker import Worker
from AquaRL.policy.ShareActorCritic import LSTMGaussianActorCritic
from AquaRL.pool.LocalPool import LocalPool
import tensorflow as tf
from model import LSTMActorCritic
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

env = gym.make("Pendulum-v1")
# observation_dims = env.observation_space.shape[0]
env = gym.make("Pendulum-v1")
observation_dims = 2
action_dims = env.action_space.shape[0]
# action_dims = 1

# tf.compat.v1.disable_eager_execution()

model_args = ModelArgs(
    using_lstm=True,
    rnn_units=32,
    num_rnn_layer=1,
    share_hidden_param=True
)

env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=4000,
    epochs=1000,
    worker_num=1,
    model_args=model_args,
    train_rnn_r2d2=True
)

hyper_parameter = PPOHyperParameters(
    # batch_size=128,
    model_args=model_args,
    c1=1,
    c2=0.001,
    update_steps=4,
    actor_critic_learning_rate=2e-3
)

actor_critic = LSTMActorCritic(size=observation_dims)

actor_critic(tf.random.normal((1, 1, 2), dtype=tf.float32))

policy = LSTMGaussianActorCritic(1, model=actor_critic, file_name='actor_critic')
data_pool = LocalPool(env_args=env_args)
ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor_critic=policy,
    works_pace='r2d2'
)


def action_fun(x):
    return 2 * x


def state_fun(x):
    x = np.squeeze(x)
    x = x[:2]
    # x = np.expand_dims(x, axis=0)
    # print(x)
    return x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy, action_fun=action_fun)

for i in range(env_args.epochs):
    worker.sample_rnn(state_fun)
    ppo.optimize()
