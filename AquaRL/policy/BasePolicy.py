import tensorflow as tf
import abc


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
