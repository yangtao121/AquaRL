import sys

sys.path.append('../')
import os
import tensorflow as tf
from mpi4py import MPI

from AquaRL.mpi.PPOMPI import PPOMPI

import gym

from AquaRL.args import PPOHyperParameters, EnvArgs
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.neural import mlp
import atexit

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# import pydevd_pycharm
#
# port_mapping = [65117, 65118, 65119, 65115]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# if rank == 0:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print("ok")

env = gym.make("LunarLanderContinuous-v2")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
env_args = EnvArgs(
    total_steps=1000,
    max_steps=1000,
    epochs=300,
    observation_dims=observation_dims,
    action_dims=action_dims,
    worker_num=size - 1
)

hyper_parameter = PPOHyperParameters(
    batch_size=128,
    update_steps=10,
    entropy_coefficient=0.01,
    clip_ratio=0.15,
    # policy_learning_rate=3e-5,
    # critic_learning_rate=5e-5
)

actor = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor',
    # output_activation='tanh'
)

value_net = mlp(
    state_dims=env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    name='value'
)

policy = GaussianPolicy(out_shape=1, model=actor, file_name='policy')
critic = CriticPolicy(model=value_net, file_name='critic')

# def action_fun(x):
#     return 2 * x


ppo = PPOMPI(hyper_parameter, policy, critic, env, comm, 'LunarLanderContinuous', env_args, action_fun=None)


@atexit.register
def clean():
    print("memory clean!")
    ppo.close_shm()


ppo.train()
ppo.close_shm()
