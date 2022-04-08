import gym
from AquaRL.worker.Worker import Worker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.policy.Discriminator import Discriminator
from AquaRL.neural import mlp
from AquaRL.pool.LocalPool import LocalPool
from AquaRL.args import PPOHyperParameters, EnvArgs, GAILParameters
from AquaRL.algo.GAIL import GAIL
from AquaRL.algo.PPO import PPO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=4000,
    epochs=150,
    worker_num=1
)

env_args_expert = EnvArgs(
    total_steps=4000,
    max_steps=200,
    epochs=1,
    observation_dims=observation_dims,
    action_dims=action_dims,
    worker_num=1
)

hyper_parameter = PPOHyperParameters(
    batch_size=128,
    update_steps=5
)

gail_parameters = GAILParameters()

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

D = mlp(
    state_dims=env_args.action_dims + env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    output_activation='sigmoid',
    name='discriminator',
)

policy = GaussianPolicy(out_shape=1, model=actor, file_name='policy')
critic = CriticPolicy(model=value_net, file_name='critic')
discriminator = Discriminator(D)

expert_policy = GaussianPolicy(action_dims)
expert_policy.load_model('policy.h5')

expert_data_pool = LocalPool(env_args=env_args_expert)

sample_worker = Worker(env, env_args_expert, expert_data_pool, expert_policy, is_training=False)

sample_worker.sample()
expert_data_pool.traj_info()
# sample_worker.INFO_sample()

data_pool = LocalPool(env_args=env_args)

worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy, is_training=True)

ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor=policy,
    critic=critic,
    discriminator=discriminator
)

gail = GAIL(gail_parameters, expert_data_pool, data_pool, discriminator)

for i in range(env_args.epochs):
    worker.sample()
    gail.optimize()
    # worker.sample()
    ppo.optimize()
