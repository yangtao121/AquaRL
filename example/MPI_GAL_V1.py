import sys

sys.path.append('../')
import os
import tensorflow as tf
from mpi4py import MPI

from AquaRL.mpi.GAILMPI import GAILPPO

import gym

from AquaRL.args import PPOHyperParameters, GAILParameters, EnvArgs
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.policy.Discriminator import Discriminator
from AquaRL.worker.Worker import Worker
from AquaRL.pool.LocalPool import LocalPool

from AquaRL.neural import mlp
import atexit

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
env_args = EnvArgs(
    max_steps=200,
    total_steps=400,
    epochs=200,
    observation_dims=observation_dims,
    action_dims=action_dims,
    worker_num=size - 1
)

env_args_expert = EnvArgs(
    total_steps=4000,
    max_steps=200,
    epochs=1,
    observation_dims=observation_dims,
    action_dims=action_dims,
    worker_num=1
)

ppo_hyper_parameter = PPOHyperParameters(
    batch_size=200,
    update_steps=10,
    entropy_coefficient=0.01
)
gail_parameters = GAILParameters()

D = mlp(
    state_dims=env_args.action_dims + env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    output_activation='sigmoid',
    name='discriminator',
)

actor = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor',
    output_activation='tanh'
)

value_net = mlp(
    state_dims=env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    name='value'
)

policy = GaussianPolicy(out_shape=1, model=actor, file_name='policy')
critic = CriticPolicy(model=value_net, file_name='critic')
discriminator = Discriminator(D)

expert_data_pool = None


def action_fun(x):
    return 2 * x


if rank == 0:
    expert_policy = GaussianPolicy(action_dims)
    expert_policy.load_model('policy.h5')
    expert_data_pool = LocalPool(env_args_expert)
    sample_worker = Worker(env, env_args_expert, expert_data_pool, expert_policy, is_training=False,
                           action_fun=action_fun)
    sample_worker.sample()
    expert_data_pool.traj_info()

comm.Barrier()
gail = GAILPPO(
    gail_parameters=gail_parameters,
    ppo_parameters=ppo_hyper_parameter,
    discriminator=discriminator,
    actor=policy,
    critic=critic,
    env=env,
    expert_data_pool=expert_data_pool,
    comm=comm,
    work_space='Pendulum_V0',
    env_args=env_args
)


@atexit.register
def clean():
    print("memory clean!")
    gail.close_shm()


gail.train()
gail.close_shm()
