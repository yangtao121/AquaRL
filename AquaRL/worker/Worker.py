from AquaRL.args import EnvArgs
import numpy as np


# TODO: is_training 替换为是否是distribution
class Worker:
    def __init__(self, env, env_args: EnvArgs, data_pool, policy, is_training=True, action_fun=None):
        """
        采样器，用于和环境互动。
        :param env: 环境。
        :param env_args: 环境参数。
        :param data_pool: 数据池。
        :param policy: 互动策略。
        :param is_training: 输出是否为分布。
        :param action_fun: 动作映射函数。
        """
        self.data_pool = data_pool
        self.policy = policy
        self.env_args = env_args
        self.env = env
        self.action_fun = action_fun
        self.is_training = is_training

        if env_args.step_training:
            self.total_run_steps = 0
            self.traj_steps = 0
            self.done = False
            self.traj_num = 0
            self.state = None
            self.sum_reward = 0
            self.reward_buffer = []
            self.trajs_lens_buffer = []

    # TODO: 合并sample函数
    def sample(self, state_function=None):
        self.data_pool.rest_pointer()

        traj_num = 0
        traj_steps = 0

        sum_reward = 0

        rewards_buffer = []
        trajs_lens_buffer = []

        state = self.env.reset()

        for i in range(self.env_args.core_steps):
            state = state.reshape(1, -1)
            if state_function is not None:
                state = state_function(state)
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

            # print(state_)

            traj_steps += 1

            sum_reward += reward

            if not done and traj_steps < self.env_args.max_steps:
                mask = 1
            else:
                mask = 0

            self.data_pool.store(state, action, reward, mask, state_.reshape(1, -1), prob)

            if not done and traj_steps < self.env_args.max_steps:
                pass
            else:
                mask = 0
                traj_num += 1
                rewards_buffer.append(sum_reward)
                trajs_lens_buffer.append(traj_steps)
                traj_steps = 0
                sum_reward = 0
                state_ = self.env.reset()

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

    def sample_rnn(self, state_function=None):
        self.data_pool.rest_pointer()

        traj_num = 0
        traj_steps = 0

        sum_reward = 0

        rewards_buffer = []
        trajs_lens_buffer = []

        state = self.env.reset()

        done = True

        for i in range(self.env_args.core_steps):
            state = state.reshape(1, -1)
            if state_function is not None:
                state = state_function(state)
            if self.is_training:
                action, prob, hidden_state, value = self.policy.get_action(state, done)
            else:
                action = self.policy.action(state)
                prob = None
                hidden_state = None
                value = None

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

            self.data_pool.store_rnn(state, action, reward, mask, state_.reshape(1, -1), prob, value, hidden_state)

            if not done and traj_steps < self.env_args.max_steps:
                pass
            else:
                done = True
                traj_num += 1
                rewards_buffer.append(sum_reward)
                trajs_lens_buffer.append(traj_steps)
                traj_steps = 0
                sum_reward = 0
                state_ = self.env.reset()

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

    def step(self):
        """
        与环境只交互一次,针对off-policy方式
        :return:
        """

        if self.total_run_steps == 0:
            self.state = self.env.reset()

        self.state = self.state.reshape(1, -1)

        if self.is_training:
            action, prob = self.policy.get_action(self.state)
        else:
            action = self.policy.action(self.state)
            prob = None

        action = np.clip(action, -1, 1)

        if self.action_fun is not None:
            action_ = self.action_fun(action)
        else:
            action_ = action

        state_, reward, done, _ = self.env.step(action_)

        self.traj_steps += 1
        self.total_run_steps += 1

        self.sum_reward += reward

        # if self.traj_steps<self.env_args.max_steps:
        #     done = False
        # else:
        #     done = True
        if not done and self.traj_steps < self.env_args.max_steps:
            mask = 1
        else:
            mask = 0

        self.data_pool.store(self.state, action, reward, mask, state_.reshape(1, -1), prob)

        if not done and self.traj_steps < self.env_args.max_steps:
            pass
        else:
            self.traj_num += 1
            self.reward_buffer.append(self.sum_reward)
            self.trajs_lens_buffer.append(self.traj_num)
            self.traj_steps = 0
            self.sum_reward = 0

            # print(self.reward_buffer)
            mean = np.mean(self.reward_buffer)
            max_reward = np.max(self.reward_buffer)
            min_reward = np.min(self.reward_buffer)

            self.data_pool.summary_trajs(
                average_reward=mean,
                max_reward=max_reward,
                min_reward=min_reward,
                average_traj_len=1,
                max_traj_len=1,
                min_traj_len=1,
                traj_num=1
            )

            self.state = self.env.reset()
            # done = True
            self.reward_buffer = []
            self.trajs_lens_buffer = []

        self.state = state_
        return done
