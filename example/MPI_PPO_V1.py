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
from AquaRL.neural import gaussian_mlp, mlp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# import pydevd_pycharm
#
# port_mapping = [65117, 65118, 65119, 65115]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

if rank == 0:
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

# print("ok")

env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
env_args = EnvArgs(
    trajs=1,
    max_steps=200,
    epochs=100,
    observation_dims=observation_dims,
    action_dims=action_dims,
    multi_worker_num=size - 1
)

hyper_parameter = PPOHyperParameters(
    batch_size=200,
    update_steps=10,
    entropy_coefficient=0.01
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

ppo = PPOMPI(hyper_parameter, policy, critic, env, comm, 'test2', env_args)
ppo.train()
ppo.close_shm()
