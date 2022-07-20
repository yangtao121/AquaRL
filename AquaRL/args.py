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

class ModelArgs:
    def __init__(self, using_lstm=False, num_rnn_layer=None, rnn_units=None):
        """

        :param using_lstm:
        :param num_rnn_layer:int
        :param rnn_units:tuple
        """
        self.using_lstm = using_lstm
        self.num_rnn_layer = num_rnn_layer
        self.rnn_units = rnn_units


class EnvArgs:
    def __init__(self, observation_dims, action_dims, max_steps, epochs, total_steps=None, worker_num=None,
                 model_args: ModelArgs = ModelArgs(),
                 buffer_size=None,
                 step_training=False,
                 train_rnn_r2d2=False,
                 ):
        """

        :param max_steps: 一个轨迹最大的步数。
        :param total_steps: 一个线程采样的最大步数
        :param worker_num: worker的数目,单线程时数目为1
        :param buffer_size: off policy时会使用此方式, 此时部分参数被屏蔽, 并且此时这个也为超参数
        """
        if worker_num is None:
            self.worker_num = 1
        else:
            self.worker_num = worker_num

        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.max_steps = max_steps

        self.model_args = model_args

        self.train_rnn_r2d2 = train_rnn_r2d2

        # self.worker_num = worker_num
        self.epochs = epochs
        self.step_training = step_training

        self.buffer_size = buffer_size

        if buffer_size is not None:
            self.total_steps = buffer_size
        else:
            self.total_steps = total_steps * self.worker_num

        if self.worker_num == 1:
            self.core_steps = total_steps
        else:
            if total_steps is not None:
                self.core_steps = total_steps
            else:
                self.core_steps = int(buffer_size / self.worker_num)

        # self.core_steps = self.total_steps

    def sync_task(self, rank):
        """
        给每个进程分配详细的存储空间，比如说起始地址。
        :param rank: 进程号
        :return:
        """
        start_step_pointer = (rank - 1) * self.core_steps
        summary_pointer = rank - 1

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


class BCParameter:
    def __init__(self, batch_size=256, learning_rate=3e-3, update_times=2, clip_ratio=0.01):
        self.learning_rate = learning_rate
        self.update_times = update_times
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio


class DDPGParameter:
    def __init__(self,
                 policy_learning_rate=0.001,
                 critic_learning_rate=0.002,
                 gamma=0.99,
                 soft_update_ratio=0.01,
                 buffer_size=5000,
                 batch_size=64
                 ):
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.soft_update_ratio = soft_update_ratio
        self.batch_size = batch_size
        self.buffer_size = buffer_size


class ActorCriticBehaviorCloningParameter:
    def __init__(self,
                 policy_learning_rate=0.001,
                 critic_learning_rate=0.002,
                 gamma=0.99,
                 soft_update_ratio=0.01,
                 batch_size=64,
                 buffer_size=5000,
                 ):
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.soft_update_ratio = soft_update_ratio
        self.batch_size = batch_size
        self.buffer_size = buffer_size


class TD3Parameter:
    def __init__(self,
                 policy_learning_rate=0.001,
                 critic_learning_rate=0.002,
                 policy_update_interval=3,
                 explore_noise_scale=1.0,  # 这个参数给policy的
                 eval_noise_scale=0.2,
                 eval_noise_clip=0.5,
                 soft_update_ratio=0.01,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=5000,
                 # offline 版本的参数
                 alpha=2.5,
                 state_normalize=False,
                 reward_normalize=False,
                 stationary_buffer=False,
                 bias=1e-3
                 ):
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.eval_noise_clip = eval_noise_clip
        self.explore_noise_scale = explore_noise_scale
        self.eval_noise_scale = eval_noise_scale
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_update_ratio = soft_update_ratio
        self.policy_update_interval = policy_update_interval
        self.buffer_size = buffer_size
        self.alpha = alpha

        self.state_normalize = state_normalize
        self.stationary_buffer = stationary_buffer
        self.reward_normalize = reward_normalize
        self.bias = bias
