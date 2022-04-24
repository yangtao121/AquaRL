import gym
from AquaRL.worker.Worker import Worker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.neural import mlp
from AquaRL.pool.LocalPool import LocalPool
from AquaRL.args import BCParameter, EnvArgs
from AquaRL.algo.BehaviorCloning import DistributionBehaviorCloning
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make("Pendulum-v0")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

expert_env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=4000,
    epochs=100,
    worker_num=1,
)

test_env_args = EnvArgs(
    observation_dims=observation_dims,
    action_dims=action_dims,
    max_steps=200,
    total_steps=2000,
    epochs=1,
    worker_num=1,
)

hyper_parameter = BCParameter(
    batch_size=256,
    learning_rate=6e-4,
    update_times=2
)

learner = mlp(
    state_dims=observation_dims,
    output_dims=action_dims,
    hidden_size=(64, 64),
    name='actor'
)

learner_policy = GaussianPolicy(out_shape=action_dims, model=learner, file_name='policy')
expert_policy = GaussianPolicy(action_dims)
expert_policy.load_model('policy.h5')

expert_data_pool = LocalPool(expert_env_args)
test_data_pool = LocalPool(test_env_args)


def action_fun(x):
    return 2 * x


expert_worker = Worker(env, expert_env_args, expert_data_pool, expert_policy, is_training=False, action_fun=action_fun)
test_worker = Worker(env, test_env_args, test_data_pool, learner_policy, is_training=False, action_fun=action_fun)

dbc = DistributionBehaviorCloning(
    hyper_parameters=hyper_parameter,
    data_pool=expert_data_pool,
    policy=learner_policy,
    workspace='dbc'
)

for i in range(expert_env_args.epochs):
    print("----------------epoch:{}--------------".format(i))
    expert_worker.sample()
    dbc._optimize()
    if (i + 1) % 5 == 0:
        test_worker.sample()
        test_data_pool.traj_info()
