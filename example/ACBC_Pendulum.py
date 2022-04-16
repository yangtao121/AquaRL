import gym
from AquaRL.args import ActorCriticBehaviorCloningParameter, EnvArgs
from AquaRL.algo.BehaviorCloning import ActorCriticBehaviorCloning

from AquaRL.worker.Worker import Worker
from AquaRL.policy.DeterminedPolicy import DeterminedPolicy
from AquaRL.policy.CriticPolicy import StateActionCriticPolicy
from AquaRL.policy.GaussianPolicy import GaussianPolicy
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

expert_env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=500000,
    epochs=10000,
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

hyper_parameter = ActorCriticBehaviorCloningParameter(
    buffer_size=expert_env_args.total_steps,
    batch_size=128,
    critic_learning_rate=6e-5,
    policy_learning_rate=3e-5,
    soft_update_ratio=0.005
)

actor = mlp(
    state_dims=test_env_args.observation_dims,
    output_dims=test_env_args.action_dims,
    hidden_size=(256, 256),
    name='actor',
    output_activation='tanh'
)

q_value_net = state_action_mlp(
    state_dims=test_env_args.observation_dims,
    action_dims=test_env_args.action_dims,
    hidden_size=(32, 256, 256),
)

expert_policy = GaussianPolicy(out_shape=action_dims, model=None, file_name='expert_policy')
expert_policy.load_model('policy.h5')

actor_policy = DeterminedPolicy(model=actor, noise=None, policy_name='policy')
critic_policy = StateActionCriticPolicy(model=q_value_net, policy_name='critic')

expert_data_pool = LocalPool(expert_env_args)
test_data_pool = LocalPool(test_env_args)

acbc = ActorCriticBehaviorCloning(
    hyper_parameter=hyper_parameter,
    actor=actor_policy,
    critic=critic_policy,
    expert_data_pool=expert_data_pool,
    test_data_pool=test_data_pool,
    work_space='ACBC'

)


def action_fun(x):
    return 2 * x


test_worker = Worker(env=env, env_args=test_env_args, data_pool=test_data_pool, policy=actor_policy, is_training=False,
                     action_fun=action_fun)
expert_worker = Worker(env=env, env_args=expert_env_args, data_pool=expert_data_pool, policy=expert_policy,
                       is_training=False, action_fun=action_fun)

expert_worker.sample()
expert_data_pool.traj_info()

for i in range(expert_env_args.epochs):
    # expert_worker.sample()
    if (i+1) % 20 == 0:
        test_worker.sample()
        print("-----------------------------")
        test_data_pool.traj_info()
        print("q loss:{}".format(q_loss))
        print('actor loss:{}'.format(actor_loss))
    q_loss, actor_loss = acbc.optimize()

test_worker.sample()
test_data_pool.traj_info()
