from AquaRL.policy.BasePolicy import BasePolicy
import tensorflow as tf
import numpy as np


class Discriminator(BasePolicy):
    def __init__(self, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)
        self.Model = model
        self.file_name = file_name

    def get_variable(self):
        return self.Model.trainable_variables

    def get_r(self, state, action):
        s_a = np.zeros(4)
        s_a[:3] = state
        s_a[3] = action
        s_a = tf.convert_to_tensor(s_a, dtype=tf.float32)
        s_a = tf.expand_dims(s_a, axis=0)
        r = self(s_a)
        r = tf.math.log(tf.clip_by_value(r, 1e-10, 1))
        return r

    # TODO: modify
    # def __call__(self, state, action):
    #     return self.Model(state, action)
