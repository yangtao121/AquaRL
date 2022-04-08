from AquaRL.pool.BasePool import BasePool
from AquaRL.args import EnvArgs
import numpy as np


class LocalPool(BasePool):
    def __init__(self, env_args: EnvArgs):
        super().__init__(env_args=env_args)
        if isinstance(self.observation_dims, tuple):
            obs_length = len(self.observation_dims)
            obs_shape = []
            obs_shape.append(self.total_steps)
            for i in range(obs_length):
                obs_shape.append(self.observation_dims[i])
            self.observation_buffer = np.zeros(obs_shape, dtype=np.float32)
            self.next_observation_buffer = np.zeros(obs_shape, dtype=np.float32)
        else:
            self.observation_buffer = np.zeros((self.total_steps, self.observation_dims), dtype=np.float32)
            self.next_observation_buffer = np.zeros((self.total_steps, self.observation_dims), dtype=np.float32)

        if isinstance(self.action_dims, tuple):
            act_length = len(self.action_dims)
            act_shape = []
            act_shape.append(self.total_steps)
            for i in range(act_length):
                act_shape.append(self.action_dims[i])
            self.action_buffer = np.zeros(act_shape, dtype=np.float32)
        else:
            self.action_buffer = np.zeros((self.total_steps, self.action_dims), dtype=np.float32)

        self.prob_buffer = np.zeros(self.action_buffer.shape, dtype=np.float32)

        self.reward_buffer = np.zeros((self.total_steps, 1))


# if __name__ == "__main__":
#     pool = LocalPool((64, 64, 1), 1, 100, 10)
#     pool.pool_info()
