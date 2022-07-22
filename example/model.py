import tensorflow as tf


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


if __name__ == '__main__':
    obs = tf.random.normal((1,3),dtype=tf.float32)
    model = ActorCritic()
    action, value = model(obs)

    print(action)
    print(value)
