import tensorflow as tf
from AquaRL.policy.BasePolicy import BasePolicy, OnlineTargetPolicy


class CriticPolicy(BasePolicy):
    def __init__(self, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)

    @tf.function
    def get_value(self, obs):
        v = self.Model(obs)
        v = tf.squeeze(v)
        return v

    def get_variable(self):
        return self.Model.trainable_variables


class StateActionCriticPolicy(OnlineTargetPolicy):
    def __init__(self, model: tf.keras.Model, policy_name=None):
        """
        当使用DDPG时候，模型的创建不要使用专家模式
        """
        super().__init__(model=model, policy_name=policy_name)

    @tf.function
    def online_value(self, state, action):
        return self.online_model([state, action])

    @tf.function
    def target_value(self, state, action):
        return self.target_model([state, action])
