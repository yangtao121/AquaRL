# 介绍

欢迎使用AquaRL强化学习框架，该框架的设计旨在让强化学习用起来更简单，更快，更方便的并行运算。AquaRL基于TensorFlow2实现，并且将网络的结构与整个框架剥离开来，只要的模型是按照keras风格写的，你都可以很方便使用该框架，而不需要进行额外的修改。这里的并行实现是通过MPI＋共享内存的形式实现。目前实现PPO和GAIL，未来将不断添加新的算法。

## 结构

![结构](../../.gitbook/assets/Code\_structure.png)

目前就放个大概的结构，后面将完整版的结构放进去。

## 未来开发计划

* [ ] 仿真环境难度自调整功能
* [x] 支持可变step的轨迹
* [ ] 完善policy定义
* [x] 添加DDPG算法
* [ ] 添加DQN算法
* [ ] 添加对循环网络的支持

