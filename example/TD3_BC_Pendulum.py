import gym
from AquaRL.args import TD3Parameter, EnvArgs
from AquaRL.algo.TD3_BC import TD3_BC

from AquaRL.worker.Worker import Worker
from AquaRL.policy.DeterminedPolicy import DeterminedPolicy
from AquaRL.policy.CriticPolicy import StateActionCriticPolicy
from AquaRL.policy.GaussianPolicy import GaussianPolicy
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
    epochs=2000,
    buffer_size=100000,
    step_training=True,
    worker_num=1
)

test_env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=2000,
    worker_num=1,
    epochs=1
)

hyper_parameter = TD3Parameter(
    buffer_size=env_args.buffer_size,
    critic_learning_rate=3e-4,
    policy_learning_rate=3e-4,
    batch_size=512,
    eval_noise_clip=0.2,
    explore_noise_scale=0.1,
    state_normalize=True,
    reward_normalize=True,
    stationary_buffer=True,
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

expert_policy = GaussianPolicy(out_shape=action_dims, model=None, file_name='expert_policy')
expert_policy.load_model('policy.h5')

data_pool = LocalPool(env_args)
test_data_pool = LocalPool(test_env_args)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=expert_policy, action_fun=action_fun,
                is_training=False)

test_worker = Worker(env=env, env_args=test_env_args, data_pool=test_data_pool, is_training=False,
                     action_fun=action_fun, policy=actor_policy)

worker.sample()
data_pool.traj_info()
data_pool.save_data('data')
data_pool.load_data('data')


def state_function(x):
    x = (x - data_pool.get_state_mean) / data_pool.get_state_std

    return x


td3_bc = TD3_BC(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    test_data_pool=test_data_pool,
    actor_policy=actor_policy,
    q_policy1=q1_policy,
    q_policy2=q2_policy,
    noise=eval_noise,
    work_space='td3_bc'
)
verbose = False

for i in range(env_args.epochs):
    verbose = False
    if (i + 1) % 20 == 0:
        test_worker.sample(state_function)
        verbose = True

    q1_loss, q2_loss, actor_loss, mse_loss = td3_bc.optimize(verbose=verbose)
