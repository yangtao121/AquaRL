from AquaRL.args import EnvArgs
import numpy as np


class Worker:
    def __init__(self, env, env_args: EnvArgs, data_pool, policy, is_training=True, action_fun=None):
        """
        采样器，用于和环境互动。
        :param env: 环境。
        :param env_args: 环境参数。
        :param data_pool: 数据池。
        :param policy: 互动策略。
        :param is_training: 是否是在训练。
        :param action_fun: 动作映射函数。
        """
        self.data_pool = data_pool
        self.policy = policy
        self.env_args = env_args
        self.env = env
        self.action_fun = action_fun
        self.is_training = is_training

    def sample(self):
        self.data_pool.rest_pointer()

        traj_num = 0
        traj_steps = 0

        sum_reward = 0

        rewards_buffer = []
        trajs_lens_buffer = []

        state = self.env.reset()

        for i in range(self.env_args.core_steps):
            state = state.reshape(1, -1)
            if self.is_training:
                action, prob = self.policy.get_action(state)
            else:
                action = self.policy.action(state)
                prob = None

            if self.action_fun is not None:
                action_ = self.action_fun(action)
            else:
                action_ = action

            state_, reward, done, _ = self.env.step(action_)

            traj_steps += 1

            sum_reward += reward

            if not done and traj_steps < self.env_args.max_steps:
                mask = 1
            else:
                mask = 0
                traj_num += 1
                rewards_buffer.append(sum_reward)
                trajs_lens_buffer.append(traj_steps)
                traj_steps = 0
                sum_reward = 0
                state_ = self.env.reset()

            self.data_pool.store(state, action, reward, mask, prob)
            state = state_

        mean = np.mean(rewards_buffer)
        max_reward = np.max(rewards_buffer)
        min_reward = np.min(rewards_buffer)
        avg_traj_len = np.average(trajs_lens_buffer)
        max_traj_len = np.max(trajs_lens_buffer)
        min_traj_len = np.min(trajs_lens_buffer)

        self.data_pool.summary_trajs(
            average_reward=mean,
            max_reward=max_reward,
            min_reward=min_reward,
            average_traj_len=avg_traj_len,
            max_traj_len=max_traj_len,
            min_traj_len=min_traj_len,
            traj_num=traj_num
        )

    # def sample(self):
    #
    #     self.data_pool.rest_pointer()
    #     for i in range(self.env_args.trajs):
    #         state = self.env.reset()
    #         sum_reward = 0
    #         # print(state)
    #
    #         for t in range(self.env_args.steps):
    #             state = state.reshape(1, -1)
    #             # print(state)
    #             action, prob = self.policy.get_action(state)
    #             # print(prob)
    #             # action = action.numpy()
    #             # prob = prob.numpy()[0]
    #             # print(prob)
    #
    #             if self.action_fun is not None:
    #                 action_ = self.action_fun(action)
    #             else:
    #                 action_ = action
    #             # action_ = action * 2
    #
    #             state_, reward, done, _ = self.env.step(action_)
    #
    #             sum_reward += reward
    #
    #             if done:
    #                 mask = 0
    #             else:
    #                 mask = 1
    #             self.data_pool.store(state, action, reward, mask, state_.reshape(1, -1), prob)
    #             state = state_
    #
    #         self.data_pool.summery_episode(sum_reward)
    # print("sub thread")
    # print(self.data_pool.prob_buffer)

# class EvaluateWorker:
#     def __init__(self, steps, trajs, env, policy):
#         self.trajs = trajs
#         self.steps = steps
#         self.env = env
#         self.policy = policy
#         self.ep_rewards = []
#         self.rewards = np.zeros((trajs, 1))
#
#     def sample(self):
#
#         for i in range(self.trajs):
#             state = self.env.reset()
#             sum_reward = 0
#             for j in range(self.steps):
#                 state = state.reshape(1, -1)
#                 # print(state)
#                 action, prob = self.policy.action(state)
#                 action_ = action * 2
#                 state_, reward, done, _ = self.env.step(action_)
#                 sum_reward += reward
#                 state = state_
#
#             self.rewards[i] = sum_reward
#
#         max_r = np.max(self.rewards)
#         min_r = np.min(self.rewards)
#         ave_r = np.mean(self.rewards)
#
#         print("Evaluate:")
#         print("Average reward:{}".format(ave_r))
#         print("Min reward:{}".format(min_r))
#         print("Max reward:{}".format(max_r))
#
#         return ave_r
#
#
# class SampleWorker:
#     def __init__(self, env, env_args: EnvArgs, data_pool, policy, action_fun=None):
#         self.data_pool = data_pool
#         self.policy = policy
#         self.env_args = env_args
#         self.env = env
#         self.action_fun = action_fun
#
#     def sample(self):
#
#         self.data_pool.rest_pointer()
#         for i in range(self.env_args.trajs):
#             state = self.env.reset()
#             sum_reward = 0
#             # print(state)
#
#             for t in range(self.env_args.steps):
#                 state = state.reshape(1, -1)
#                 # print(state)
#                 action = self.policy.action(state)
#
#                 if self.action_fun is not None:
#                     action_ = self.action_fun(action)
#                 else:
#                     action_ = action
#
#                 state_, reward, done, _ = self.env.step(action_)
#
#                 sum_reward += reward
#
#                 if done:
#                     mask = 0
#                 else:
#                     mask = 1
#                 self.data_pool.store(state, action, reward, mask, state_.reshape(1, -1), 1)
#                 state = state_
#
#             self.data_pool.summery_episode(sum_reward)
#
#     def INFO_sample(self):
#         print("Average reward:{}".format(self.data_pool.get_average_reward))
#         print("Min reward:{}".format(self.data_pool.get_min_reward))
#         print("Max reward:{}".format(self.data_pool.get_max_reward))
#
#
# class GAILWorker:
#     def __init__(self, env, env_args: EnvArgs, data_pool, policy, discriminator):
#         self.data_pool = data_pool
#         self.policy = policy
#         self.env_args = env_args
#         self.env = env
#         self.discriminator = discriminator
#
#     def sample(self):
#
#         self.data_pool.rest_pointer()
#         for i in range(self.env_args.trajs):
#             state = self.env.reset()
#             sum_reward = 0
#             # print(state)
#
#             for t in range(self.env_args.steps):
#                 state = state.reshape(1, -1)
#                 # print(state)
#                 action, prob = self.policy.get_action(state)
#
#                 action_ = action * 2
#
#                 state_, reward, done, _ = self.env.step(action_)
#
#                 d_reward = self.discriminator.get_r(state, action)
#                 # print(d_reward)
#
#                 sum_reward += reward
#
#                 if done:
#                     mask = 0
#                 else:
#                     mask = 1
#                 self.data_pool.store(state, action, d_reward, mask, state_.reshape(1, -1), prob)
#                 state = state_
#
#             self.data_pool.summery_episode(sum_reward)
