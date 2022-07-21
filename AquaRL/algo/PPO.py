from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import PPOHyperParameters
import tensorflow_probability as tfp


# TODO: clear data shape
class PPO(BaseAlgo):
    def __init__(self, hyper_parameters: PPOHyperParameters, data_pool, actor=None,
                 critic=None, works_pace=None, discriminator=None):
        super().__init__(hyper_parameters, data_pool, works_pace)
        self.actor = actor
        self.critic = critic

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)
        self.discriminator = discriminator

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        # print(self.data_pool.next_observation_buffer)
        tf_next_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.next_observation_buffer)

        tf_values_buffer = self.critic.get_value(tf_observation_buffer)
        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        tf_old_probs = self.data_pool.convert_to_tensor(self.data_pool.prob_buffer)
        tf_next_values_buffer = self.critic.get_value(tf_next_observation_buffer)
        #
        # tf_mask_buffer = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer)
        if self.discriminator is None:
            tf.reward_buffer = self.data_pool.convert_to_tensor(self.data_pool.reward_buffer)
        else:
            tf.reward_buffer = tf.math.log(
                self.discriminator.get_rewards_buffer(tf_observation_buffer, tf_action_buffer))

        gae, target = self.cal_gae_target(self.data_pool.reward_buffer, tf_values_buffer.numpy(),tf_next_values_buffer.numpy(),
                                          self.data_pool.mask_buffer)
        tf_gae = self.data_pool.convert_to_tensor(gae)
        tf_target = self.data_pool.convert_to_tensor(target)

        max_steps = self.data_pool.total_steps
        # print(max_steps)

        critic_loss, actor_loss, surrogate_loss, entropy_loss = self.cal_loss(tf_observation_buffer, tf_action_buffer,
                                                                              tf_gae, tf_target, tf_old_probs)

        print("Training before:")
        print("Critic loss:{}".format(critic_loss))
        print("Actor loss:{}".format(actor_loss))
        print("Surrogate loss:{}".format(surrogate_loss))
        print("Entropy loss:{}".format(entropy_loss))

        with self.before_summary_writer.as_default():
            tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
            tf.summary.scalar('PPO/actor_loss', actor_loss, self.epoch)
            tf.summary.scalar('PPO/surrogate loss', surrogate_loss, self.epoch)
            tf.summary.scalar('PPO/entropy_loss', entropy_loss, self.epoch)

        for _ in tf.range(0, self.hyper_parameters.update_steps):
            start_pointer = 0
            end_pointer = self.hyper_parameters.batch_size - 1
            while end_pointer <= max_steps - 1:
                state = tf_observation_buffer[start_pointer: end_pointer]
                action = tf_action_buffer[start_pointer: end_pointer]
                gae = tf_gae[start_pointer: end_pointer]
                target = tf_target[start_pointer: end_pointer]
                old_prob = tf_old_probs[start_pointer: end_pointer]

                self.train_actor(state, action, gae, old_prob)
                self.train_critic(state, target)
                # print(acc_loss)
                # print(ok)
                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

        critic_loss, actor_loss, surrogate_loss, entropy_loss = self.cal_loss(tf_observation_buffer, tf_action_buffer,
                                                                              tf_gae, tf_target, tf_old_probs)

        print("Training after:")
        print("Critic loss:{}".format(critic_loss))
        print("Actor loss:{}".format(actor_loss))
        print("Surrogate loss:{}".format(surrogate_loss))
        print("Entropy loss:{}".format(entropy_loss))

        with self.after_summary_writer.as_default():
            tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
            tf.summary.scalar('PPO/actor_loss', actor_loss, self.epoch)
            tf.summary.scalar('PPO/surrogate loss', surrogate_loss, self.epoch)
            tf.summary.scalar('PPO/entropy_loss', entropy_loss, self.epoch)

    @tf.function
    def train_critic(self, observation, target):
        """
        inputs are tf.tensor.
        :param observation:
        :param target:
        :return: tensor
        """
        if self.hyper_parameters.clip_critic_value:
            with tf.GradientTape() as tape:
                v = self.critic(observation)
                surrogate1 = tf.square(v[1:] - target[1:])
                surrogate2 = tf.square(
                    tf.clip_by_value(v[1:], v[:-1] - self.hyper_parameters.clip_ratio,
                                     v[:-1] + self.hyper_parameters.clip_ratio) - target[1:])
                critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            with tf.GradientTape() as tape:
                v = self.critic(observation)
                critic_loss = tf.reduce_mean(tf.square(target - v))

        grad = tape.gradient(critic_loss, self.critic.get_variable())
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.get_variable()))

        return critic_loss

    @tf.function
    def train_actor(self, state, action, advantage, old_prob):
        # TODO:Add new function entropy loss.
        """
        inputs are tf.tensor
        :param state:
        :param action:
        :param advantage:
        :param old_prob:
        :return:
        """
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(state)
            pi = tfp.distributions.Normal(mu, sigma)

            # print(action.shape)
            # print(old_prob.shape)
            new_prob = tf.clip_by_value(pi.prob(action), 1e-6, 1)
            ratio = new_prob / old_prob
            surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                     1 + self.hyper_parameters.clip_ratio) * advantage
                )
            )
            entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

            loss = -(entropy_loss * self.hyper_parameters.entropy_coefficient + surrogate_loss)
        actor_grad = tape.gradient(loss, self.actor.get_variable())
        # print(actor_grad)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.get_variable()))

        return loss

    @tf.function
    def cal_loss(self, state, action, advantage, target, old_prob):
        if self.hyper_parameters.clip_critic_value:
            v = self.critic(state)
            surrogate1 = tf.square(v[1:] - target[1:])
            surrogate2 = tf.square(
                tf.clip_by_value(v[1:], v[:-1] - self.hyper_parameters.clip_ratio,
                                 v[:-1] + self.hyper_parameters.clip_ratio) - target[1:])
            critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            v = self.critic(state)
            critic_loss = tf.reduce_mean(tf.square(target - v))

        mu, sigma = self.actor(state)
        pi = tfp.distributions.Normal(mu, sigma)
        new_prob = tf.clip_by_value(pi.prob(action), 1e-6, 1)
        ratio = new_prob / old_prob
        surrogate_loss = tf.reduce_mean(
            tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                 1 + self.hyper_parameters.clip_ratio) * advantage
            )
        )
        entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

        actor_loss = -(entropy_loss + surrogate_loss)

        return critic_loss, -actor_loss, surrogate_loss, entropy_loss

    def write_parameter(self):
        with self.main_summary_writer.as_default():
            clip_ratio = 'clip ratio:{}'.format(self.hyper_parameters.clip_ratio)
            actor_learning_rate = 'actor learning rate:{}'.format(self.hyper_parameters.policy_learning_rate)
            critic_learning_rate = 'critic learning rate:{}'.format(self.hyper_parameters.critic_learning_rate)
            batch_size = 'batch size:{}'.format(self.hyper_parameters.batch_size)
            update_times = 'update times:{}'.format(self.hyper_parameters.update_steps)
            gamma = 'gamma:{}'.format(self.hyper_parameters.gamma)
            lambada = 'lambada:{}'.format(self.hyper_parameters.lambada)
            tolerance = 'tolerance:{}'.format(self.hyper_parameters.tolerance)
            entropy_coefficient = 'entropy coefficient:{}'.format(self.hyper_parameters.entropy_coefficient)
            reward_scale = 'use reward scale:{}'.format(self.hyper_parameters.reward_scale)
            center_adv = 'use center adv:{}'.format(self.hyper_parameters.center_adv)
            tf.summary.text('PPO_parameter', clip_ratio, step=self.epoch)
            tf.summary.text('PPO_parameter', actor_learning_rate, step=self.epoch)
            tf.summary.text('PPO_parameter', critic_learning_rate, step=self.epoch)
            tf.summary.text('PPO_parameter', batch_size, step=self.epoch)
            tf.summary.text('PPO_parameter', update_times, step=self.epoch)
            tf.summary.text('PPO_parameter', gamma, step=self.epoch)
            tf.summary.text('PPO_parameter', lambada, step=self.epoch)
            tf.summary.text('PPO_parameter', tolerance, step=self.epoch)
            tf.summary.text('PPO_parameter', entropy_coefficient, step=self.epoch)
            tf.summary.text('PPO_parameter', reward_scale, step=self.epoch)
            tf.summary.text('PPO_parameter', center_adv, step=self.epoch)
