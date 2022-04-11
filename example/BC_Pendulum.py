import gym
from AquaRL.worker.Worker import Worker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.BCPolicy import BCPolicy
from AquaRL.neural import mlp
from AquaRL.pool.LocalPool import LocalPool
from AquaRL.args import BCParameter, EnvArgs
from AquaRL.algo.BehaviorCloning import BehaviorCloning
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
    epochs=100,
    worker_num=1
)

hyper_parameter = BCParameter(
    batch_size=256,
    learning_rate=6e-4,
    update_times=2
)

learner = mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor'
)

learner_policy = BCPolicy(model=learner, file_name='policy')
expert_policy = GaussianPolicy(action_dims)
expert_policy.load_model('policy.h5')

data_pool = LocalPool(env_args)


def action_fun(x):
    return 2 * x


sample_worker = Worker(env, env_args, data_pool, expert_policy, is_training=False, action_fun=action_fun)
evaluator = Worker(env, env_args, data_pool, learner_policy, is_training=False, action_fun=action_fun)

bc = BehaviorCloning(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    policy=learner_policy,
    workspace='BC_pendulum'
)

for i in range(env_args.epochs):
    sample_worker.sample()
    bc.optimize()
    if i % 10 == 0:
        evaluator.sample()
        data_pool.traj_info()
