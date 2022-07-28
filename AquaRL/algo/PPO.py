from AquaRL.algo.BaseAlgo import BaseAlgo
import tensorflow as tf
from AquaRL.args import PPOHyperParameters
import tensorflow_probability as tfp
import numpy as np


# TODO: clear data shape
class PPO(BaseAlgo):
    def __init__(self, hyper_parameters: PPOHyperParameters, data_pool, actor=None,
                 critic=None, actor_critic=None, works_pace=None, discriminator=None):
        super().__init__(hyper_parameters, data_pool, works_pace)

        if self.hyper_parameters.model_args.share_hidden_param:
            self.actor_critic = actor_critic
            self.actor_critic_optimizer = tf.optimizers.Adam(
                learning_rate=self.hyper_parameters.actor_critic_learning_rate)
        else:
            self.actor = actor
            self.critic = critic

            self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.critic_learning_rate)
            self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.hyper_parameters.policy_learning_rate)

        self.discriminator = discriminator

    def _optimize(self):
        tf_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.observation_buffer)
        # print(self.data_pool.next_observation_buffer)
        tf_next_observation_buffer = self.data_pool.convert_to_tensor(self.data_pool.next_observation_buffer)

        tf_action_buffer = self.data_pool.convert_to_tensor(self.data_pool.action_buffer)
        tf_old_probs = self.data_pool.convert_to_tensor(self.data_pool.prob_buffer)

        if not self.hyper_parameters.model_args.share_hidden_param:
            tf_values_buffer = self.critic.get_value(tf_observation_buffer)
            tf_next_values_buffer = self.critic.get_value(tf_next_observation_buffer)
        else:
            tf_values_buffer = self.actor_critic.get_value(tf_observation_buffer)
            tf_next_values_buffer = self.actor_critic.get_value(tf_next_observation_buffer)
        # tf_mask_buffer = self.data_pool.convert_to_tensor(self.data_pool.mask_buffer)
        if self.discriminator is None:
            tf.reward_buffer = self.data_pool.convert_to_tensor(self.data_pool.reward_buffer)
        else:
            tf.reward_buffer = tf.math.log(
                self.discriminator.get_rewards_buffer(tf_observation_buffer, tf_action_buffer))

        gae, target = self.cal_gae_target(self.data_pool.reward_buffer, tf_values_buffer.numpy(),
                                          tf_next_values_buffer.numpy(),
                                          self.data_pool.mask_buffer)
        tf_gae = self.data_pool.convert_to_tensor(gae)
        tf_target = self.data_pool.convert_to_tensor(target)

        max_steps = self.data_pool.total_steps
        # print(max_steps)

        if not self.hyper_parameters.model_args.share_hidden_param:
            critic_loss, actor_loss, surrogate_loss, entropy_loss = self.cal_loss(tf_observation_buffer,
                                                                                  tf_action_buffer,
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

            critic_loss, actor_loss, surrogate_loss, entropy_loss = self.cal_loss(tf_observation_buffer,
                                                                                  tf_action_buffer,
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

        else:
            critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss = self.cal_actor_critic_loss(
                tf_observation_buffer,
                tf_action_buffer,
                tf_gae, tf_target, tf_old_probs)

            print("Training before:")
            print("Critic loss:{}".format(critic_loss))
            print("Actor loss:{}".format(actor_surrogate_loss))
            print("Entropy loss:{}".format(actor_entropy_loss))
            print("Total loss:{}".format(total_loss))

            with self.before_summary_writer.as_default():
                tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
                tf.summary.scalar('PPO/actor_loss', actor_surrogate_loss, self.epoch)
                tf.summary.scalar('PPO/entropy_loss', actor_entropy_loss, self.epoch)
                tf.summary.scalar('PPO/total loss', total_loss, self.epoch)

            for _ in tf.range(0, self.hyper_parameters.update_steps):
                start_pointer = 0
                end_pointer = self.hyper_parameters.batch_size - 1
                while end_pointer <= max_steps - 1:
                    state = tf_observation_buffer[start_pointer: end_pointer]
                    action = tf_action_buffer[start_pointer: end_pointer]
                    gae = tf_gae[start_pointer: end_pointer]
                    target = tf_target[start_pointer: end_pointer]
                    old_prob = tf_old_probs[start_pointer: end_pointer]

                    self.train_actor_critic(state, action, gae, target, old_prob)
                    # print(acc_loss)
                    # print(ok)
                    start_pointer = end_pointer
                    end_pointer = end_pointer + self.hyper_parameters.batch_size
            critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss = self.cal_actor_critic_loss(
                tf_observation_buffer,
                tf_action_buffer,
                tf_gae, tf_target, tf_old_probs)
            print("Training after:")
            print("Critic loss:{}".format(critic_loss))
            print("Actor loss:{}".format(actor_surrogate_loss))
            print("Entropy loss:{}".format(actor_entropy_loss))
            print("Total loss:{}".format(total_loss))

            with self.after_summary_writer.as_default():
                tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
                tf.summary.scalar('PPO/actor_loss', actor_surrogate_loss, self.epoch)
                tf.summary.scalar('PPO/entropy_loss', actor_entropy_loss, self.epoch)
                tf.summary.scalar('PPO/total loss', total_loss, self.epoch)

    def _optimize_r2d2(self):

        next_values = np.zeros_like(self.data_pool.value_buffer)
        next_values[:-1] = self.data_pool.value_buffer[1:]

        gae, target = self.cal_gae_target((self.data_pool.reward_buffer + 8) / 8, self.data_pool.value_buffer,
                                          next_values,
                                          self.data_pool.mask_buffer)

        trajs_obs, burn_ins_obs, trajs_act, trajs_reward, trajs_next_obs, trajs_prob, trajs_value, trajs_hidden, trajs_gaes, trajs_targets = self.data_pool.r2d2_data_process(
            gae, target)
        tf_traj_hidden = self.data_pool.convert_to_tensor(trajs_hidden)

        if self.hyper_parameters.model_args.using_lstm:
            tf_hidden1, tf_hidden2 = tf.split(tf_traj_hidden, 2, axis=1)
            tf_hidden = (tf_hidden1, tf_hidden2)
        else:
            tf_hidden = tf_traj_hidden

        tf_trajs_obs = self.data_pool.convert_to_tensor(trajs_obs)
        tf_burn_ins_obs = self.data_pool.convert_to_tensor(burn_ins_obs)
        tf_trajs_act = self.data_pool.convert_to_tensor(trajs_act)
        tf_trajs_prob = self.data_pool.convert_to_tensor(trajs_prob)
        tf_trajs_gae = self.data_pool.convert_to_tensor(trajs_gaes)
        tf_trajs_target = self.data_pool.convert_to_tensor(trajs_targets)

        critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss = self.cal_actor_critic_r2d2_loss(
            traj_obs=tf_trajs_obs, traj_burn_in=tf_burn_ins_obs, traj_act=tf_trajs_act, traj_adv=tf_trajs_gae,
            traj_target=tf_trajs_target, traj_prob=tf_trajs_prob, traj_hidden=tf_hidden)

        print("Training before:")
        print("Critic loss:{}".format(critic_loss))
        print("Actor loss:{}".format(actor_surrogate_loss))
        print("Entropy loss:{}".format(actor_entropy_loss))
        print("Total loss:{}".format(total_loss))

        with self.before_summary_writer.as_default():
            tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
            tf.summary.scalar('PPO/actor_loss', actor_surrogate_loss, self.epoch)
            tf.summary.scalar('PPO/entropy_loss', actor_entropy_loss, self.epoch)
            tf.summary.scalar('PPO/total loss', total_loss, self.epoch)

        for _ in tf.range(0, self.hyper_parameters.update_steps):
            start_pointer = 0
            end_pointer = self.hyper_parameters.batch_size
            max_step = trajs_obs.shape[0]
            while end_pointer <= max_step - 1:
                batch_traj_obs = tf_trajs_obs[start_pointer: end_pointer]
                batch_traj_burn_in = tf_burn_ins_obs[start_pointer: end_pointer]
                batch_traj_act = tf_trajs_act[start_pointer: end_pointer]
                batch_traj_gae = tf_trajs_gae[start_pointer: end_pointer]
                batch_traj_target = tf_trajs_target[start_pointer: end_pointer]
                batch_traj_prob = tf_trajs_prob[start_pointer: end_pointer]
                if self.hyper_parameters.model_args.using_lstm:
                    batch_traj_hidden1 = tf_hidden1[start_pointer: end_pointer]
                    batch_traj_hidden2 = tf_hidden2[start_pointer: end_pointer]
                    batch_traj_hidden = (batch_traj_hidden1, batch_traj_hidden2)
                else:
                    batch_traj_hidden = tf_hidden[start_pointer: end_pointer]

                self.train_actor_critic_r2d2(
                    traj_obs=batch_traj_obs, traj_burn_in=batch_traj_burn_in, traj_act=batch_traj_act,
                    traj_adv=batch_traj_gae,
                    traj_target=batch_traj_target, traj_prob=batch_traj_prob, traj_hidden=batch_traj_hidden)

                start_pointer = end_pointer
                end_pointer = end_pointer + self.hyper_parameters.batch_size

        critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss = self.cal_actor_critic_r2d2_loss(
            traj_obs=tf_trajs_obs, traj_burn_in=tf_burn_ins_obs, traj_act=tf_trajs_act, traj_adv=tf_trajs_gae,
            traj_target=tf_trajs_target, traj_prob=tf_trajs_prob, traj_hidden=tf_hidden)

        print("Training after:")
        print("Critic loss:{}".format(critic_loss))
        print("Actor loss:{}".format(actor_surrogate_loss))
        print("Entropy loss:{}".format(actor_entropy_loss))
        print("Total loss:{}".format(total_loss))

        with self.after_summary_writer.as_default():
            tf.summary.scalar('PPO/critic_loss', critic_loss, self.epoch)
            tf.summary.scalar('PPO/actor_loss', actor_surrogate_loss, self.epoch)
            tf.summary.scalar('PPO/entropy_loss', actor_entropy_loss, self.epoch)
            tf.summary.scalar('PPO/total loss', total_loss, self.epoch)

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
    def train_actor_critic(self, state, action, advantage, target, old_prob):
        with tf.GradientTape() as tape:
            # mu, sigma = self.actor_critic.actor(state)
            mu, v, sigma = self.actor_critic.get_actor_critic(state)
            pi = tfp.distributions.Normal(mu, sigma)
            new_prob = tf.clip_by_value(pi.prob(action), 1e-6, 1)
            ratio = new_prob / old_prob

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                     1 + self.hyper_parameters.clip_ratio
                                     ) * advantage
                )
            )

            if self.hyper_parameters.clip_critic_value:
                # v = self.actor_critic.critic(state)
                surrogate1 = tf.square(v[1:] - target[1:])
                surrogate2 = tf.square(
                    tf.clip_by_value(v[1:], v[:-1] - self.hyper_parameters.clip_ratio,
                                     v[:-1] + self.hyper_parameters.clip_ratio) - target[1:])
                critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            else:
                # v = self.actor_critic.critic(state)
                critic_loss = tf.reduce_mean(tf.square(target - v))

            actor_entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

            total_loss = -(
                    actor_surrogate_loss - self.hyper_parameters.c1 * critic_loss + self.hyper_parameters.c2 * actor_entropy_loss)
        grad = tape.gradient(total_loss, self.actor_critic.get_variable())
        self.actor_critic_optimizer.apply_gradients(zip(grad, self.actor_critic.get_variable()))

        return total_loss

    @tf.function
    def train_actor_critic_r2d2(self, traj_obs, traj_burn_in, traj_act, traj_adv, traj_target, traj_prob, traj_hidden):
        with tf.GradientTape() as tape:
            # 初始化lstm参数
            self.actor_critic.Model(traj_burn_in, traj_hidden, training=False)

            mu, v, sigma = self.actor_critic.get_actor_critic(traj_obs, training=True)
            pi = tfp.distributions.Normal(mu, sigma)
            new_prob = tf.clip_by_value(pi.prob(traj_act), 1e-6, 1)
            ratio = new_prob / traj_prob

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * traj_adv,
                    tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                     1 + self.hyper_parameters.clip_ratio
                                     ) * traj_adv
                )
            )

            critic_loss = tf.reduce_mean(tf.square(traj_target - v))

            actor_entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

            total_loss = -(
                    actor_surrogate_loss - self.hyper_parameters.c1 * critic_loss + self.hyper_parameters.c2 * actor_entropy_loss)
        grad = tape.gradient(total_loss, self.actor_critic.get_variable())

        self.actor_critic_optimizer.apply_gradients(zip(grad, self.actor_critic.get_variable()))

        return total_loss

    @tf.function
    def cal_actor_critic_r2d2_loss(self, traj_obs, traj_burn_in, traj_act, traj_adv, traj_target, traj_prob,
                                   traj_hidden):
        _, _, hidde = self.actor_critic.Model(traj_burn_in, traj_hidden, training=False)

        # print(hidde)

        mu, v, sigma = self.actor_critic.get_actor_critic(traj_obs, training=False)
        pi = tfp.distributions.Normal(mu, sigma)
        new_prob = tf.clip_by_value(pi.prob(traj_act), 1e-6, 1)
        ratio = new_prob / traj_prob

        actor_surrogate_loss = tf.reduce_mean(
            tf.minimum(
                ratio * traj_adv,
                tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                 1 + self.hyper_parameters.clip_ratio
                                 ) * traj_adv
            )
        )

        critic_loss = tf.reduce_mean(tf.square(traj_target - v))

        actor_entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

        total_loss = -(
                actor_surrogate_loss - self.hyper_parameters.c1 * critic_loss + self.hyper_parameters.c2 * actor_entropy_loss)

        return critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss

    @tf.function
    def cal_actor_critic_loss(self, state, action, advantage, target, old_prob):
        mu, sigma = self.actor_critic.actor(state)
        pi = tfp.distributions.Normal(mu, sigma)
        new_prob = tf.clip_by_value(pi.prob(action), 1e-6, 1)
        ratio = new_prob / old_prob

        actor_surrogate_loss = tf.reduce_mean(
            tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - self.hyper_parameters.clip_ratio,
                                 1 + self.hyper_parameters.clip_ratio
                                 ) * advantage
            )
        )

        if self.hyper_parameters.clip_critic_value:
            v = self.actor_critic.critic(state)
            surrogate1 = tf.square(v[1:] - target[1:])
            surrogate2 = tf.square(
                tf.clip_by_value(v[1:], v[:-1] - self.hyper_parameters.clip_ratio,
                                 v[:-1] + self.hyper_parameters.clip_ratio) - target[1:])
            critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            v = self.actor_critic.critic(state)
            critic_loss = tf.reduce_mean(tf.square(target - v))

        actor_entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

        total_loss = actor_surrogate_loss - self.hyper_parameters.c1 * critic_loss + self.hyper_parameters.c2 * actor_entropy_loss

        return critic_loss, actor_surrogate_loss, actor_entropy_loss, total_loss

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

    def optimize(self):
        # print(self.data_pool.prob_buffer)

        with self.average_summary_writer.as_default():
            tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_average_reward, step=self.epoch)
        with self.max_summary_writer.as_default():
            tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_max_reward, step=self.epoch)
        with self.min_summary_writer.as_default():
            tf.summary.scalar("Traj_Info/Reward", self.data_pool.get_min_reward, step=self.epoch)

        mean_len = self.data_pool.get_average_traj_len
        max_len = self.data_pool.get_max_traj_len
        min_len = self.data_pool.get_min_traj_len

        if max_len == max_len:
            pass
        else:
            with self.average_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Len", mean_len, step=self.epoch)
            with self.max_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Len", max_len, step=self.epoch)
            with self.min_summary_writer.as_default():
                tf.summary.scalar("Traj_Info/Len", min_len, step=self.epoch)

        print("_______________epoch:{}____________________".format(self.epoch))
        self.data_pool.traj_info()

        # self.data_pool.reward_buffer = (self.data_pool.reward_buffer + 8) / 8
        if self.hyper_parameters.model_args.using_lstm:
            self._optimize_r2d2()
        else:
            self._optimize()

        self.epoch += 1
