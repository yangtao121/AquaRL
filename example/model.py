import tensorflow as tf
import numpy as np


class ActorCritic(tf.keras.Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.share_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(64, activation='relu'),
             ]
        )

        self.actor_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(32, activation='relu'),
             tf.keras.layers.Dense(1, activation='tanh')]
        )

        self.critic_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(32, activation='relu'),
             tf.keras.layers.Dense(1)
             ]
        )

    @tf.function
    def __call__(self, x):
        share_obs = self.share_model(x)
        action = self.actor_model(share_obs)
        value = self.critic_model(share_obs)

        return action, value

    def actor(self, obs):
        share_obs = self.share_model(obs)
        action = self.actor_model(share_obs)
        return action

    def critic(self, obs):
        share_obs = self.share_model(obs)
        value = self.critic_model(share_obs)

        return value


class LSTMActorCritic(tf.keras.Model):
    def __init__(self):
        super(LSTMActorCritic, self).__init__()

        self.share_lstm = tf.keras.layers.LSTM(32, input_shape=(3,), return_sequences=True, return_state=True)

        self.actor_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(1, activation='tanh')]
        )

        self.critic_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(1)
             ]
        )

        self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32), dtype=tf.float32))

    # @tf.function
    def call(self, obs, initial_state=None, training=False, done=False):
        """
        该函数用于optimize过程
        :param done:
        :param inputs:(batchsize,time, obs)
        :param initial_state:tuple (last_seq, hidden_state)
        :param training:
        :return:
        """
        if initial_state is not None:
            self.hidden_state = initial_state
            # self.hidden_state = hidden_state
        # else:
        #     self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32), dtype=tf.float32))

        if done:
            self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32), dtype=tf.float32))

        whole_seq, last_seq, hidden_state = self.share_lstm(obs, self.hidden_state, training=training)
        hidden_state = (last_seq, hidden_state)

        self.hidden_state = hidden_state

        action, value = self.call_actor_critic(whole_seq, training=training)

        return action, value, hidden_state

    def _actor(self, obs, hidden_state):
        whole_seq, last_seq, hidden_state = self.share_lstm(obs, hidden_state)

        action = self.actor_model(last_seq)

        return action, hidden_state

    def actor(self, obs, done):

        if done:
            self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32)))

        action, hidden_state = self._actor(obs, self.hidden_state)

        self.hidden_state = hidden_state

        return action, np.hstack(hidden_state)

    def critic(self, obs):
        share_obs = self.share_model(obs)
        value = self.critic_model(share_obs)

        return value

    @tf.function
    def call_actor_critic(self, whole_seq, training):
        action = self.actor_model(whole_seq, training=training)
        value = self.critic_model(whole_seq, training=training)

        return action, value


if __name__ == '__main__':
    obs = tf.random.normal((2, 1, 2), dtype=tf.float32)
    model = LSTMActorCritic()
    action, value, hidden_state = model(obs)

    print(tf.squeeze(action))
    print(value)
    print(hidden_state)
