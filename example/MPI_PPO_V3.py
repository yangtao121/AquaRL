import sys

sys.path.append('../')
import os
import tensorflow as tf
from mpi4py import MPI

from AquaRL.mpi.PPOMPI import PPOMPI

import gym

from AquaRL.args import PPOHyperParameters, ModelArgs, EnvArgs
from AquaRL.policy.ShareActorCritic import LSTMGaussianActorCritic
from model import LSTMActorCritic
import atexit

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# import pydevd_pycharm
#
# port_mapping = [65117, 65118, 65119, 65115]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

# env = gym.make("LunarLanderContinuous-v2")
env = gym.make("Pendulum-v1")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

model_args = ModelArgs(
    using_lstm=True,
    rnn_units=32,
    num_rnn_layer=1,
    share_hidden_param=True,
    r2d2=True,
    traj_len=80,
    over_lap_size=30
)
env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=300,
    total_steps=600,
    epochs=1000,
    worker_num=size - 1,
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

actor_critic = LSTMActorCritic()

actor_critic(tf.random.normal((1, 1, 3), dtype=tf.float32))

policy = LSTMGaussianActorCritic(1, model=actor_critic, file_name='actor_critic')


def action_fun(x):
    return 2 * x


ppo = PPOMPI(
    hyper_parameters=hyper_parameter,
    env=env,
    comm=comm,
    env_args=env_args,
    actor_critic=policy,
    action_fun=action_fun,
    work_space='r2d2_PPO4'
)


@atexit.register
def clean():
    print("memory clean!")
    ppo.close_shm()


ppo.train()
ppo.close_shm()
