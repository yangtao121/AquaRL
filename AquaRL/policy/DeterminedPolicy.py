import tensorflow as tf
from AquaRL.policy.BasePolicy import OnlineTargetPolicy


class DeterminedPolicy(OnlineTargetPolicy):
    def __init__(self, model: tf.keras.Model, noise, policy_name=None):
        super().__init__(model=model, policy_name=policy_name)
        self.noise = noise

    @tf.function
    def online_action(self, state):
        return self.online_model(state)

    @tf.function
    def target_action(self, state):
        return self.target_model(state)

    @tf.function
    def __call__(self, state):
        return self.online_action(state)

    def action(self, state):
        mu = self(state)

        action = mu + self.noise()

        return action
