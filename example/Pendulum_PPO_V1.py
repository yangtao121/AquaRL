import gym
from AquaRL.args import PPOHyperParameters, EnvArgs
from AquaRL.algo.PPO import PPO

from AquaRL.worker.Worker import Worker, EvaluateWorker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.neural import gaussian_mlp, mlp
from AquaRL.pool.LocalPool import LocalPool

env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

env_args = EnvArgs(
    trajs=18,
    max_steps=200,
    epochs=100,
    observation_dims=observation_dims,
    action_dims=action_dims
)

hyper_parameter = PPOHyperParameters(
    batch_size=200
)

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

policy = GaussianPolicy(out_shape=1, model=actor, file_name='policy')
critic = CriticPolicy(model=value_net, file_name='critic')

data_pool = LocalPool(observation_dims, action_dims, env_args.total_steps, env_args.trajs)

ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor=policy,
    critic=critic
)
worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy)
evaluate_worker = EvaluateWorker(200, 10, env, policy)
max_mark = -1e6
for i in range(env_args.epochs):
    worker.sample()
    ppo.optimize()
    if i > 50:
        mark = evaluate_worker.sample()
        # if mark > max_mark:
        #     max_mark = mark
        #     policy.save_model('actor.h5')

# policy.save_model('actor.h5')
