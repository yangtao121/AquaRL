import tensorflow as tf
import abc


# TODO: 这个地方需要重新设计提升代码可重复使用特点
class BasePolicy(abc.ABC):
    def __init__(self, model=None, file_name=None):
        self.Model = model
        self.file_name = file_name

    def __call__(self, obs):
        return self.Model(obs)

    def save_model(self, file=None):
        if file is None:
            tf.keras.models.save_model(self.Model, self.file_name + '.h5', overwrite=True)
        else:
            tf.keras.models.save_model(self.Model, filepath=file + '/' + self.file_name + '.h5', overwrite=True)

    def save_weights(self, file=None):
        if file is None:
            self.Model.save_weights(self.file_name + '.h5', overwrite=True)
        else:
            self.Model.save_weights(filepath=file + '/' + self.file_name + '.h5', overwrite=True)

        print('save complete.')

    def load_weights(self, file=None):
        self.Model.load_weights(filepath=file + '/' + self.file_name + '.h5')
        # print('load complete.')

    def load_model(self, file):
        self.Model = tf.keras.models.load_model(filepath=file, compile=False)

    def net_visual(self, file=None):
        self.Model.summary()
        if file is None:
            tf.keras.utils.plot_model(self.Model, self.file_name + '.png', show_shapes=True)
        else:
            tf.keras.utils.plot_model(self.Model, file, show_shapes=True)

    @abc.abstractmethod
    def get_variable(self):
        """
        获取模型的可训练参数
        """


# TODO: 所有的policy风格按这个这个选择
class OnlineTargetPolicy(abc.ABC):
    def __init__(self, model: tf.keras.Model, policy_name=None):
        self.online_model = model
        self.target_model = tf.keras.models.clone_model(model)
        self.policy_name = policy_name

    def soft_update(self, tau):
        new_weights = []
        target_weight = self.target_model.weights

        for i, weight in enumerate(self.online_model.weights):
            new_weights.append(target_weight[i] * (1 - tau) + tau * weight)

        self.target_model.set_weights(new_weights)

    # @property
    def get_variable(self):
        return self.online_model.trainable_variables

    def save_model(self, path=None):
        tf.keras.models.save_model(self.online_model, path + '/' + self.policy_name + '_online_model.h5')
        tf.keras.models.save_model(self.target_model, path + '/' + self.policy_name + '_target_model.h5')

    def load_model(self, path=None):
        tf.keras.models.load_model(path + '/' + self.policy_name + '_online_model.h5')
        tf.keras.models.load_model(path + '/' + self.policy_name + '_target_model.h5')
