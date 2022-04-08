from multiprocessing import shared_memory
from AquaRL.pool.BasePool import BasePool
from AquaRL.args import EnvArgs
import numpy as np


class MainThreadSharaMemery(BasePool):
    def __init__(self, env_args: EnvArgs, name):
        super().__init__(env_args)
        self.pool_name = name
        if isinstance(self.observation_dims, tuple):
            obs_length = len(self.observation_dims)
            obs_shape = []
            obs_shape.append(self.total_steps)
            for i in range(obs_length):
                obs_shape.append(self.observation_dims[i])
            observation_share = np.zeros(obs_shape, dtype=np.float32)
        else:
            observation_share = np.zeros((self.total_steps, self.observation_dims), dtype=np.float32)
        if isinstance(self.action_dims, tuple):
            act_length = len(self.action_dims)
            act_shape = []
            act_shape.append(self.total_steps)
            for i in range(act_length):
                act_shape.append(self.action_dims[i])
            action_share = np.zeros(act_shape, dtype=np.float32)
        else:
            action_share = np.zeros((self.total_steps, self.action_dims), dtype=np.float32)

        reward_share = np.zeros((self.total_steps, 1), dtype=np.int32)
        prob_share = np.zeros(action_share.shape, dtype=np.float32)
        mask_share = np.zeros((self.total_steps, 1), dtype=np.float32)

        # create share memery
        self.shm_observation = shared_memory.SharedMemory(create=True, size=observation_share.nbytes,
                                                          name=name + "_observation")

        self.shm_action = shared_memory.SharedMemory(create=True, size=action_share.nbytes, name=name + "_action")
        self.shm_prob = shared_memory.SharedMemory(create=True, size=prob_share.nbytes, name=name + "_prob")
        self.shm_reward = shared_memory.SharedMemory(create=True, size=reward_share.nbytes, name=name + "_reward")
        self.shm_mask = shared_memory.SharedMemory(create=True, size=mask_share.nbytes, name=name + "_mask")

        self.shm_average_reward = shared_memory.SharedMemory(create=True, size=self.average_rewards_buffer.nbytes,
                                                             name=name + "_ave_rewards")
        self.shm_max_reward = shared_memory.SharedMemory(create=True, size=self.max_rewards_buffer.nbytes,
                                                         name=name + "_max_rewards")
        self.shm_min_reward = shared_memory.SharedMemory(create=True, size=self.min_rewards_buffer.nbytes,
                                                         name=name + "_min_rewards")

        self.shm_ave_traj_len = shared_memory.SharedMemory(create=True, size=self.average_traj_len_buffer.nbytes,
                                                           name=name + "_ave_traj_len")
        self.shm_max_traj_len = shared_memory.SharedMemory(create=True, size=self.max_traj_len_buffer.nbytes,
                                                           name=name + "_max_traj_len")
        self.shm_min_traj_len = shared_memory.SharedMemory(create=True, size=self.min_traj_len_buffer.nbytes,
                                                           name=name + "_min_traj_len")
        self.shm_traj_num = shared_memory.SharedMemory(create=True, size=self.traj_num_buffer.nbytes,
                                                       name=name + "_traj_num")

        # create a NumPy array backed by shared memory
        self.observation_buffer = np.ndarray(observation_share.shape, dtype=np.float32, buffer=self.shm_observation.buf)

        self.action_buffer = np.ndarray(action_share.shape, dtype=np.float32, buffer=self.shm_action.buf)
        self.reward_buffer = np.ndarray(reward_share.shape, dtype=np.float32, buffer=self.shm_reward.buf)
        self.prob_buffer = np.ndarray(prob_share.shape, dtype=np.float32, buffer=self.shm_prob.buf)
        self.mask_buffer = np.ndarray(mask_share.shape, dtype=np.float32, buffer=self.shm_mask.buf)
        self.average_rewards_buffer = np.ndarray(self.average_rewards_buffer.shape, dtype=np.float32,
                                                 buffer=self.shm_average_reward.buf)
        self.max_rewards_buffer = np.ndarray(self.max_rewards_buffer.shape, dtype=np.float32,
                                             buffer=self.shm_max_reward.buf)
        self.min_rewards_buffer = np.ndarray(self.min_rewards_buffer.shape, dtype=np.float32,
                                             buffer=self.shm_min_reward.buf)
        self.average_traj_len_buffer = np.ndarray(self.average_traj_len_buffer.shape, dtype=np.float32,
                                                  buffer=self.shm_ave_traj_len.buf)
        self.max_traj_len_buffer = np.ndarray(self.max_traj_len_buffer.shape, dtype=np.float32,
                                              buffer=self.shm_max_traj_len.buf)
        self.min_traj_len_buffer = np.ndarray(self.min_traj_len_buffer.shape, dtype=np.float32,
                                              buffer=self.shm_min_traj_len.buf)
        self.traj_num_buffer = np.ndarray(self.traj_num_buffer.shape, dtype=np.float32, buffer=self.shm_traj_num.buf)

    def close_shm(self):
        del self.observation_buffer
        del self.action_buffer
        del self.prob_buffer
        del self.reward_buffer
        del self.mask_buffer

        self.shm_observation.close()
        self.shm_action.close()
        self.shm_prob.close()
        self.shm_reward.close()
        self.shm_mask.close()

        self.shm_observation.unlink()
        self.shm_reward.unlink()
        self.shm_action.unlink()
        self.shm_prob.unlink()
        self.shm_mask.unlink()


class SubThreadShareMemery(BasePool):
    def __init__(self, env_args: EnvArgs, rank, name):
        super().__init__(env_args)

        self.rank = rank

        self.pool_name = name
        if isinstance(self.observation_dims, tuple):
            obs_length = len(self.observation_dims)
            obs_shape = []
            obs_shape.append(self.total_steps)
            for i in range(obs_length):
                obs_shape.append(self.observation_dims[i])
            observation_share = np.zeros(obs_shape, dtype=np.float32)
        else:
            observation_share = np.zeros((self.total_steps, self.observation_dims), dtype=np.float32)

        if isinstance(self.action_dims, tuple):
            act_length = len(self.action_dims)
            act_shape = []
            act_shape.append(self.total_steps)
            for i in range(act_length):
                act_shape.append(self.action_dims[i])
            action_share = np.zeros(act_shape, dtype=np.float32)
        else:
            action_share = np.zeros((self.total_steps, self.action_dims), dtype=np.float32)

        reward_share = np.zeros((self.total_steps, 1))
        prob_share = np.zeros(action_share.shape, dtype=np.float32)
        mask_share = np.zeros((self.total_steps, 1), dtype=np.float32)

        self.shm_observation = shared_memory.SharedMemory(name=name + '_observation')
        self.shm_action = shared_memory.SharedMemory(name=name + "_action")
        self.shm_prob = shared_memory.SharedMemory(name=name + "_prob")
        self.shm_reward = shared_memory.SharedMemory(name=name + "_reward")
        self.shm_mask = shared_memory.SharedMemory(name=name + "_mask")

        self.shm_average_reward = shared_memory.SharedMemory(name=name + "_ave_rewards")
        self.shm_max_reward = shared_memory.SharedMemory(name=name + "_max_rewards")
        self.shm_min_reward = shared_memory.SharedMemory(name=name + "_min_rewards")

        self.shm_ave_traj_len = shared_memory.SharedMemory(name=name + "_ave_traj_len")
        self.shm_max_traj_len = shared_memory.SharedMemory(name=name + "_max_traj_len")
        self.shm_min_traj_len = shared_memory.SharedMemory(name=name + "_min_traj_len")
        self.shm_traj_num = shared_memory.SharedMemory(name=name + "_traj_num")

        # create a NumPy array backed by shared memory
        self.observation_buffer = np.ndarray(observation_share.shape, dtype=np.float32, buffer=self.shm_observation.buf)
        self.action_buffer = np.ndarray(action_share.shape, dtype=np.float32, buffer=self.shm_action.buf)
        self.reward_buffer = np.ndarray(reward_share.shape, dtype=np.float32, buffer=self.shm_reward.buf)
        self.prob_buffer = np.ndarray(prob_share.shape, dtype=np.float32, buffer=self.shm_prob.buf)
        self.mask_buffer = np.ndarray(mask_share.shape, dtype=np.float32, buffer=self.shm_mask.buf)

        self.average_rewards_buffer = np.ndarray(self.average_rewards_buffer.shape, dtype=np.float32,
                                                 buffer=self.shm_average_reward.buf)
        self.max_rewards_buffer = np.ndarray(self.max_rewards_buffer.shape, dtype=np.float32,
                                             buffer=self.shm_max_reward.buf)
        self.min_rewards_buffer = np.ndarray(self.min_rewards_buffer.shape, dtype=np.float32,
                                             buffer=self.shm_min_reward.buf)
        self.average_traj_len_buffer = np.ndarray(self.average_traj_len_buffer.shape, dtype=np.float32,
                                                  buffer=self.shm_ave_traj_len.buf)
        self.max_traj_len_buffer = np.ndarray(self.max_traj_len_buffer.shape, dtype=np.float32,
                                              buffer=self.shm_max_traj_len.buf)
        self.min_traj_len_buffer = np.ndarray(self.min_traj_len_buffer.shape, dtype=np.float32,
                                              buffer=self.shm_min_traj_len.buf)
        self.traj_num_buffer = np.ndarray(self.traj_num_buffer.shape, dtype=np.float32, buffer=self.shm_traj_num.buf)

        self.cnt = 0
        self.start_pointer, self.summary_start_pinter = env_args.sync_task(rank)
        self.summary_pointer = self.summary_start_pinter
        self.pointer = self.start_pointer

    def rest_pointer(self):
        self.pointer = self.start_pointer
        self.cnt = 0

    def close_shm(self):
        del self.observation_buffer
        del self.action_buffer
        del self.prob_buffer
        del self.reward_buffer
        del self.mask_buffer
        self.shm_observation.close()
        self.shm_action.close()
        self.shm_prob.close()
        self.shm_reward.close()
        self.shm_mask.close()

# if __name__ == "__main__":
#     pool = SubThreadShareMemery((64, 64, 1), 3, 200, 49, 50, 'test1')
#     # print(pool.mask_buffer)
#     for _ in range(60):
#         obs = np.ones((64, 64, 1), dtype=np.float32)
#         action = np.ones(3)
#         reward = 1
#         mask = 1
#         pool.store(obs, action, reward, mask, obs, action)
#
#     pool.close_shm()
