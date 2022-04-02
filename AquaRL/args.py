class EnvArgs:
    def __init__(self, trajs, max_steps, epochs, observation_dims, action_dims,
                 multi_worker_num=None):
        if multi_worker_num is not None:
            self.total_steps = trajs * max_steps * multi_worker_num
            self.multi_worker_num = multi_worker_num
        else:
            self.total_steps = trajs * max_steps
            self.multi_worker_num = 1

        self.trajs = trajs
        self.steps = max_steps

        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.epochs = epochs


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
