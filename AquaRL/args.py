# class EnvArgs:
#     def __init__(self, trajs, max_steps, epochs, observation_dims, action_dims,
#                  multi_worker_num=None):
#         if multi_worker_num is not None:
#             self.total_steps = trajs * max_steps * multi_worker_num
#             self.multi_worker_num = multi_worker_num
#             self.thread_step = trajs * max_steps
#             self.total_trajs = trajs * multi_worker_num
#         else:
#             self.total_steps = trajs * max_steps
#             self.multi_worker_num = 1
#             self.thread_step = self.total_steps
#
#         self.trajs = trajs
#
#         self.steps = max_steps
#
#         self.observation_dims = observation_dims
#         self.action_dims = action_dims
#         self.epochs = epochs


class EnvArgs:
    def __init__(self, observation_dims, action_dims, max_steps, total_steps, epochs, worker_num):
        """

        :param max_steps: 一个轨迹最大的步数。
        :param total_steps: 一个线程采样的最大步数
        :param worker_num: worker的数目,单线程时数目为1
        """
        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.max_steps = max_steps
        self.core_steps = total_steps
        self.total_steps = total_steps*worker_num
        self.worker_num = worker_num
        self.epochs = epochs

    def sync_task(self, rank):
        """
        给每个进程分配详细的存储空间，比如说起始地址。
        :param rank: 进程号
        :return:
        """
        start_step_pointer = (rank - 1) * self.core_steps
        summary_pointer = rank-1

        return start_step_pointer, summary_pointer


class PPOHyperParameters:
    def __init__(self,
                 clip_ratio=0.2,
                 policy_learning_rate=3e-4,
                 critic_learning_rate=1e-3,
                 batch_size=32,
                 update_steps=10,
                 gamma=0.99,
                 lambada=0.95,
                 tolerance=1e-6,
                 entropy_coefficient=0,
                 reward_scale=False,
                 scale=False,
                 center=False,
                 center_adv=False,
                 clip_critic_value=False
                 ):
        """
        center in reward scale
        """
        self.clip_ratio = clip_ratio
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_steps = update_steps
        self.gamma = gamma
        self.lambada = lambada
        self.tolerance = tolerance
        self.center = center
        self.clip_critic_value = clip_critic_value
        self.reward_scale = reward_scale
        self.scale = scale
        self.center_adv = center_adv
        self.batch_size = batch_size
        self.entropy_coefficient = entropy_coefficient


class GAILParameters:
    def __init__(self, learning_rate=3e-3, update_times=2):
        self.learning_rate = learning_rate
        self.update_times = update_times
