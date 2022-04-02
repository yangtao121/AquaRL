import tensorflow as tf
from AquaRL.policy.BasePolicy import BasePolicy


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
