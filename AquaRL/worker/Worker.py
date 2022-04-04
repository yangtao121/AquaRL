from AquaRL.args import EnvArgs
import numpy as np


class Worker:
    def __init__(self, env, env_args: EnvArgs, data_pool, policy):
        self.data_pool = data_pool
        self.policy = policy
        self.env_args = env_args
        self.env = env

    def sample(self):

        self.data_pool.rest_pointer()
        for i in range(self.env_args.trajs):
            state = self.env.reset()
            sum_reward = 0
            # print(state)

            for t in range(self.env_args.steps):
                state = state.reshape(1, -1)
                # print(state)
                action, prob = self.policy.get_action(state)
                # print(prob)
                # action = action.numpy()
                # prob = prob.numpy()[0]
                # print(prob)

                action_ = action * 2

                state_, reward, done, _ = self.env.step(action_)

                sum_reward += reward

                if done:
                    mask = 0
                else:
                    mask = 1
                self.data_pool.store(state, action, reward, mask, state_.reshape(1, -1), prob)
                state = state_

            self.data_pool.summery_episode(sum_reward)
        # print("sub thread")
        # print(self.data_pool.prob_buffer)


class EvaluateWorker:
    def __init__(self, steps, trajs, env, policy):
        self.trajs = trajs
        self.steps = steps
        self.env = env
        self.policy = policy
        self.ep_rewards = []
        self.rewards = np.zeros((trajs, 1))

    def sample(self):

        for i in range(self.trajs):
            state = self.env.reset()
            sum_reward = 0
            for j in range(self.steps):
                state = state.reshape(1, -1)
                # print(state)
                action, prob = self.policy.action(state)
                action_ = action * 2
                state_, reward, done, _ = self.env.step(action_)
                sum_reward += reward
                state = state_

            self.rewards[i] = sum_reward

        max_r = np.max(self.rewards)
        min_r = np.min(self.rewards)
        ave_r = np.mean(self.rewards)

        print("Evaluate:")
        print("Average reward:{}".format(ave_r))
        print("Min reward:{}".format(min_r))
        print("Max reward:{}".format(max_r))

        return ave_r


class SampleWorker:
    def __init__(self, env, env_args: EnvArgs, data_pool, policy):
        self.data_pool = data_pool
        self.policy = policy
        self.env_args = env_args
        self.env = env

    def sample(self):

        self.data_pool.rest_pointer()
        for i in range(self.env_args.trajs):
            state = self.env.reset()
            sum_reward = 0
            # print(state)

            for t in range(self.env_args.steps):
                state = state.reshape(1, -1)
                # print(state)
                action = self.policy.action(state)

                action_ = action * 2

                state_, reward, done, _ = self.env.step(action_)

                sum_reward += reward

                if done:
                    mask = 0
                else:
                    mask = 1
                self.data_pool.store(state, action, reward, mask, state_.reshape(1, -1), 1)
                state = state_

            self.data_pool.summery_episode(sum_reward)

    def INFO_sample(self):
        print("Average reward:{}".format(self.data_pool.get_average_reward))
        print("Min reward:{}".format(self.data_pool.get_min_reward))
        print("Max reward:{}".format(self.data_pool.get_max_reward))


class GAILWorker:
    def __init__(self, env, env_args: EnvArgs, data_pool, policy, discriminator):
        self.data_pool = data_pool
        self.policy = policy
        self.env_args = env_args
        self.env = env
        self.discriminator = discriminator

    def sample(self):

        self.data_pool.rest_pointer()
        for i in range(self.env_args.trajs):
            state = self.env.reset()
            sum_reward = 0
            # print(state)

            for t in range(self.env_args.steps):
                state = state.reshape(1, -1)
                # print(state)
                action, prob = self.policy.get_action(state)

                action_ = action * 2

                state_, reward, done, _ = self.env.step(action_)

                d_reward = self.discriminator.get_r(state, action)
                # print(d_reward)

                sum_reward += reward

                if done:
                    mask = 0
                else:
                    mask = 1
                self.data_pool.store(state, action, d_reward, mask, state_.reshape(1, -1), prob)
                state = state_

            self.data_pool.summery_episode(sum_reward)