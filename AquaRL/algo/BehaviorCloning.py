from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import BCParameter

"""
This is a beta version.
输入是observation，输出是action
"""


class BehaviorCloning(BaseAlgo):
    """
    action是label.
    """

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            learning_rate = 'learning rate:{}'.format(self.hyper_parameters.learning_rate)
            update_times = 'update times:{}'.format(self.hyper_parameters.update_times)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)

            tf.summary.text('BC_parameter', learning_rate, step=self.epoch)
            tf.summary.text('BC_parameter', update_times, step=self.epoch)
            tf.summary.text('BC_parameter', batch_size, step=self.epoch)

    def __init__(self, hyper_parameters: BCParameter, data_pool, policy, workspace):
        super().__init__(hyper_parameters=hyper_parameters, data_pool=data_pool, work_space=workspace)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.learning_rate)
        self.policy = policy

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        max_steps = self.data_pool.total_steps

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training before:")
        print("MSE loss:{}".format(loss))
        with self.before_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

        for _ in tf.range(0, self.hyper_parameters.update_times):
            start_pointer = 0
            end_pointer = self.hyper_parameters.batch_size - 1

            while end_pointer <= max_steps:
                state = tf_observation_buffer[start_pointer: end_pointer]
                action = tf_action_buffer[start_pointer: end_pointer]

                self.train_policy(state, action)

                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

        loss = self.cal_loss(tf_observation_buffer, tf_action_buffer)
        print("Training after:")
        print("MSE loss:{}".format(loss))
        with self.after_summary_writer.as_default():
            tf.summary.scalar('BC/loss', loss, self.epoch)

    @tf.function
    def train_policy(self, observation, action):
        with tf.GradientTape() as tape:
            prediction = self.policy(observation)
            mse = tf.reduce_mean(tf.square(prediction - action))
        grad = tape.gradient(mse, self.policy.get_variable())
        self.policy_optimizer.apply_gradients(zip(grad, self.policy.get_variable()))

        return mse

    # @tf.function
    def cal_loss(self, observation, action):
        prediction = self.policy(observation)
        # print(prediction[0])
        # print(action)
        mse = tf.reduce_mean(tf.square(prediction - action))

        return mse
