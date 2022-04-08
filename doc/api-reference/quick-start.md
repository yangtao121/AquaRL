---
description: AquaRL目前还在完善之中，每个版本改动都可能比较大。
---

# Quick Start

## 安装教程

Aqua RL主要依赖于Tensorflow2，版本最好为2.4，python版本为3.8。

```bash
conda install tensroflow-gpu
conda install -c conda-forge mpi4py
pip install 'tensorflow-probability<0.13'
pip install gym
```

从github上克隆代码包并放到您的工作文件夹上。

```bash
git clone https://github.com/yangtao121/AquaRL
```

## 使用教程

这里以PPO训练Pendulum-V0为例子：

### 单线程使用：

#### Step1:  导入依赖项

```python
import gym
from AquaRL.args import PPOHyperParameters, EnvArgs
from AquaRL.algo.PPO import PPO
from AquaRL.worker.Worker import Worker, EvaluateWorker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
from AquaRL.policy.CriticPolicy import CriticPolicy
from AquaRL.neural import gaussian_mlp, mlp
from AquaRL.pool.LocalPool import LocalPool
```

`AquaRL.args`里面包含所使用算法的超参数配置以及环境参数配置；`AquaRL.algo`优化算法；`AquaRL.woker`采样器，用来收集数据；`AquaRL.pool` 用来存储数据和通信功能；`AquaRL.policy`用来定义对网络数据如何处理。网络结构可以任意，输出最好为-1~1之间，为keras模型即可。

#### Step2：环境初始化，参数选择

```Python
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

hyper_parameter = PPOHyperParameters(
    batch_size=200
)
```

#### Step3：policy的实现

```python
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
```

#### Step4：创建算法并训练

```python
data_pool = LocalPool(env_args=env_args)

ppo = PPO(
    hyper_parameters=hyper_parameter,
    data_pool=data_pool,
    actor=policy,
    critic=critic
)


def action_fun(x):
    return 2 * x


worker = Worker(env=env, env_args=env_args, data_pool=data_pool, policy=policy, action_fun=action_fun)
max_mark = -1e6
for i in range(env_args.epochs):
    worker.sample()
    ppo.optimize()
```

``action_fun``用来定义输入处理函数。
