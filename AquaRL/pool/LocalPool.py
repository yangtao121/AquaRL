from AquaRL.pool.BasePool import BasePool
import numpy as np


class LocalPool(BasePool):
    def __init__(self, observation_dims, action_dims, max_steps, epochs):
        """

        :param observation_dims: example for image: (w,h,channel)
        :param action_dims:
        :param max_steps:
        """
        super().__init__(observation_dims, action_dims, max_steps, epochs)
        if isinstance(observation_dims, tuple):
            obs_length = len(observation_dims)
            obs_shape = []
            obs_shape.append(max_steps)
            for i in range(obs_length):
                obs_shape.append(observation_dims[i])
            self.observation_buffer = np.zeros(obs_shape, dtype=np.float32)
            self.next_observation_buffer = np.zeros(obs_shape, dtype=np.float32)
        else:
            self.observation_buffer = np.zeros((max_steps, observation_dims), dtype=np.float32)
            self.next_observation_buffer = np.zeros((max_steps, observation_dims), dtype=np.float32)

        if isinstance(action_dims, tuple):
            act_length = len(action_dims)
            act_shape = []
            act_shape.append(max_steps)
            for i in range(act_length):
                act_shape.append(action_dims[i])
            self.action_buffer = np.zeros(act_shape, dtype=np.float32)
        else:
            self.action_buffer = np.zeros((max_steps, action_dims), dtype=np.float32)

        self.prob_buffer = np.zeros(self.action_buffer.shape, dtype=np.float32)

        self.reward_buffer = np.zeros((max_steps, 1))

        self.episode_reward_buffer = np.zeros((epochs, 1), dtype=np.float32)


if __name__ == "__main__":
    pool = LocalPool((64, 64, 1), 1, 100, 10)
    pool.pool_info()
