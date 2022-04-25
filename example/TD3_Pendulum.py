import gym
from AquaRL.args import TD3Parameter, EnvArgs
from AquaRL.algo.TD3 import TD3

from AquaRL.worker.Worker import Worker
from AquaRL.policy.DeterminedPolicy import DeterminedPolicy
from AquaRL.policy.CriticPolicy import StateActionCriticPolicy
from AquaRL.neural import mlp, state_action_mlp

from AquaRL.pool.LocalPool import LocalPool
import tensorflow as tf
from AquaRL.noise.GaussianNoise import GaussianNoise
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
    epochs=300,
    buffer_size=50000,
    step_training=True,
    worker_num=1
)

hyper_parameter = TD3Parameter(
    buffer_size=env_args.buffer_size,
    critic_learning_rate=3e-4,
    policy_learning_rate=3e-4,
    batch_size=128
)

actor = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(256, 256),
    name='actor',
    output_activation='tanh'
)

q1_value_net = state_action_mlp(
    state_dims=env_args.observation_dims,
    action_dims=1,
    hidden_size=(32, 256, 256),
)

q2_value_net = state_action_mlp(
    state_dims=env_args.observation_dims,
    action_dims=1,
    hidden_size=(32, 256, 256),
)

explore_noise = GaussianNoise(action_dims, 0, hyper_parameter.explore_noise_scale)
eval_noise = GaussianNoise(action_dims, 0, hyper_parameter.eval_noise_scale)

actor_policy = DeterminedPolicy(model=actor, noise=explore_noise, policy_name='actor')
q1_policy = StateActionCriticPolicy(model=q1_value_net, policy_name='q1_net')
q2_policy = StateActionCriticPolicy(model=q2_value_net, policy_name='q2_net')

data_pool = LocalPool(env_args)

td3 = TD3(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor_policy=actor_policy,
    q_policy1=q1_policy,
    q_policy2=q2_policy,
    noise=eval_noise,
    work_space='td3'
)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=actor_policy, action_fun=action_fun,
                is_training=True)

done = False

for i in range(env_args.epochs):
    # noise.reset()
    data_pool.rest_pointer()
    done = False
    while not done:
        done = worker.step()
        if i > 3:
            q1_loss, q2_loss, actor_loss = td3.optimize()

    if i > 3:
        print("q1 loss:{}".format(q1_loss))
        print("q2 loss:{}".format(q2_loss))
        print('actor loss:{}'.format(actor_loss))
