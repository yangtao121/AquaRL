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

`AquaRL.args`里面包含所使用算法的超参数配置以及环境参数配置

