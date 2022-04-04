from multiprocessing import shared_memory
import numpy as np
from AquaRL.pool.BasePool import BasePool


class MainThreadSharaMemery(BasePool):
    def __init__(self, observation_dims, action_dims, max_steps, epochs, total_traj ,name):
        super().__init__(observation_dims, action_dims, max_steps, epochs)
        self.pool_name = name
        if isinstance(observation_dims, tuple):
            obs_length = len(observation_dims)
            obs_shape = []
            obs_shape.append(max_steps)
            for i in range(obs_length):
                obs_shape.append(observation_dims[i])
            observation_share = np.zeros(obs_shape, dtype=np.float32)
            next_observation_share = np.zeros(obs_shape, dtype=np.float32)
        else:
            observation_share = np.zeros((max_steps, observation_dims), dtype=np.float32)
            next_observation_share = np.zeros((max_steps, observation_dims), dtype=np.float32)

        if isinstance(action_dims, tuple):
            act_length = len(action_dims)
            act_shape = []
            act_shape.append(max_steps)
            for i in range(act_length):
                act_shape.append(action_dims[i])
            action_share = np.zeros(act_shape, dtype=np.float32)
        else:
            action_share = np.zeros((max_steps, action_dims), dtype=np.float32)

        reward_share = np.zeros((max_steps, 1), dtype=np.int32)
        prob_share = np.zeros(action_share.shape, dtype=np.float32)

        # advantage_share = np.zeros((max_steps, 1), dtype=np.float32)
        # target_share = np.zeros((max_steps, 1), dtype=np.float32)

        mask_share = np.zeros((max_steps, 1), dtype=np.float32)
        episode_reward_share = np.zeros((total_traj, 1), dtype=np.float32)

        # create share memery
        self.shm_observation = shared_memory.SharedMemory(create=True, size=observation_share.nbytes,
                                                          name=name + "_observation")
        self.shm_next_observation = shared_memory.SharedMemory(create=True, size=next_observation_share.nbytes,
                                                               name=name + "_next_observation")
        self.shm_action = shared_memory.SharedMemory(create=True, size=action_share.nbytes, name=name + "_action")
        self.shm_prob = shared_memory.SharedMemory(create=True, size=prob_share.nbytes, name=name + "_prob")
        self.shm_reward = shared_memory.SharedMemory(create=True, size=reward_share.nbytes, name=name + "_reward")
        self.shm_mask = shared_memory.SharedMemory(create=True, size=mask_share.nbytes, name=name + "_mask")
        self.shm_episode_reward = shared_memory.SharedMemory(create=True, size=episode_reward_share.nbytes,
                                                             name=name + "_episode_reward")

        # create a NumPy array backed by shared memory
        self.observation_buffer = np.ndarray(observation_share.shape, dtype=np.float32, buffer=self.shm_observation.buf)
        self.next_observation_buffer = np.ndarray(next_observation_share.shape, dtype=np.float32,
                                                  buffer=self.shm_next_observation.buf)
        self.action_buffer = np.ndarray(action_share.shape, dtype=np.float32, buffer=self.shm_action.buf)
        self.reward_buffer = np.ndarray(reward_share.shape, dtype=np.float32, buffer=self.shm_reward.buf)
        self.prob_buffer = np.ndarray(prob_share.shape, dtype=np.float32, buffer=self.shm_prob.buf)
        self.mask_buffer = np.ndarray(mask_share.shape, dtype=np.float32, buffer=self.shm_mask.buf)
        self.episode_reward_buffer = np.ndarray(episode_reward_share.shape, dtype=np.float32,
                                                buffer=self.shm_episode_reward.buf)

    def close_shm(self):
        del self.observation_buffer
        del self.next_observation_buffer
        del self.action_buffer
        del self.prob_buffer
        del self.reward_buffer
        del self.mask_buffer
        del self.episode_reward_buffer

        self.shm_next_observation.close()
        self.shm_observation.close()
        self.shm_action.close()
        self.shm_prob.close()
        self.shm_reward.close()
        self.shm_mask.close()
        self.shm_episode_reward.close()

        self.shm_observation.unlink()
        self.shm_next_observation.unlink()
        self.shm_reward.unlink()
        self.shm_action.unlink()
        self.shm_prob.unlink()
        self.shm_mask.unlink()
        self.shm_episode_reward.unlink()


class SubThreadShareMemery(BasePool):
    def __init__(self, observation_dims, action_dims, max_steps, start_pointer, steps, epochs, epoch_start_pointer,total_traj,
                 name):
        super().__init__(observation_dims, action_dims, max_steps, epochs)

        self.pool_name = name
        if isinstance(observation_dims, tuple):
            obs_length = len(observation_dims)
            obs_shape = []
            obs_shape.append(max_steps)
            for i in range(obs_length):
                obs_shape.append(observation_dims[i])
            observation_share = np.zeros(obs_shape, dtype=np.float32)
            next_observation_share = np.zeros(obs_shape, dtype=np.float32)
        else:
            observation_share = np.zeros((max_steps, observation_dims), dtype=np.float32)
            next_observation_share = np.zeros((max_steps, observation_dims), dtype=np.float32)

        if isinstance(action_dims, tuple):
            act_length = len(action_dims)
            act_shape = []
            act_shape.append(max_steps)
            for i in range(act_length):
                act_shape.append(action_dims[i])
            action_share = np.zeros(act_shape, dtype=np.float32)
        else:
            action_share = np.zeros((max_steps, action_dims), dtype=np.float32)

        reward_share = np.zeros((max_steps, 1))
        prob_share = np.zeros(action_share.shape, dtype=np.float32)
        mask_share = np.zeros((max_steps, 1), dtype=np.float32)
        episode_reward_share = np.zeros((total_traj, 1), dtype=np.float32)
        # print(name)

        self.shm_observation = shared_memory.SharedMemory(name=name + '_observation')
        self.shm_next_observation = shared_memory.SharedMemory(name=name + "_next_observation")
        self.shm_action = shared_memory.SharedMemory(name=name + "_action")
        self.shm_prob = shared_memory.SharedMemory(name=name + "_prob")
        self.shm_reward = shared_memory.SharedMemory(name=name + "_reward")
        self.shm_mask = shared_memory.SharedMemory(name=name + "_mask")
        self.shm_episode_reward = shared_memory.SharedMemory(name=name + "_episode_reward")

        # create a NumPy array backed by shared memory
        self.observation_buffer = np.ndarray(observation_share.shape, dtype=np.float32, buffer=self.shm_observation.buf)
        self.next_observation_buffer = np.ndarray(next_observation_share.shape, dtype=np.float32,
                                                  buffer=self.shm_next_observation.buf)
        self.action_buffer = np.ndarray(action_share.shape, dtype=np.float32, buffer=self.shm_action.buf)
        self.reward_buffer = np.ndarray(reward_share.shape, dtype=np.float32, buffer=self.shm_reward.buf)
        self.prob_buffer = np.ndarray(prob_share.shape, dtype=np.float32, buffer=self.shm_prob.buf)
        self.mask_buffer = np.ndarray(mask_share.shape, dtype=np.float32, buffer=self.shm_mask.buf)
        self.episode_reward_buffer = np.ndarray(episode_reward_share.shape, dtype=np.float32,
                                                buffer=self.shm_episode_reward.buf)

        self.star_pointer = start_pointer
        self.pointer = start_pointer
        self.steps = steps
        self.cnt = 0
        self.epoch_start_pointer = epoch_start_pointer
        self.episode_pointer = self.epoch_start_pointer

    # def _store(self, observation, action, reward, mask, next_observation, prob):
    #
    #     self.observation_buffer[self.pointer] = observation
    #     self.next_observation_buffer[self.pointer] = next_observation
    #     self.action_buffer[self.pointer] = action
    #     self.reward_buffer[self.pointer] = reward
    #     self.mask_buffer[self.pointer] = mask
    #     self.prob_buffer[self.pointer] = prob
    #
    #     self.pointer += 1
    #     self.cnt += 1
    #     if self.cnt > self.steps:
    #         raise RuntimeError("Beyond maximum boundary: step buffer")

    def rest_pointer(self):
        self.pointer = self.star_pointer
        self.cnt = 0
        self.episode_pointer = self.epoch_start_pointer

    # def summery_episode(self, episode_reward):
    #     self.episode_reward_buffer[self.episode_pointer] = episode_reward
    #     self.episode_pointer += 1
    #     if self.episode_pointer - self.epoch_start_pointer > self.

    def close_shm(self):
        del self.observation_buffer
        del self.next_observation_buffer
        del self.action_buffer
        del self.prob_buffer
        del self.reward_buffer
        del self.mask_buffer
        del self.episode_reward_buffer

        self.shm_next_observation.close()
        self.shm_observation.close()
        self.shm_action.close()
        self.shm_prob.close()
        self.shm_reward.close()
        self.shm_mask.close()
        self.shm_episode_reward.close()


if __name__ == "__main__":
    pool = SubThreadShareMemery((64, 64, 1), 3, 200, 49, 50, 'test1')
    # print(pool.mask_buffer)
    for _ in range(60):
        obs = np.ones((64, 64, 1), dtype=np.float32)
        action = np.ones(3)
        reward = 1
        mask = 1
        pool.store(obs, action, reward, mask, obs, action)

    pool.close_shm()
