from AquaRL.policy.BasePolicy import BasePolicy


class BCPolicy(BasePolicy):
    def __init__(self, model=None, file_name=None):
        super().__init__(model=model, file_name=file_name)

    def get_variable(self):
        return self.Model.trainable_variables

    def action(self, obs):
        return self.Model(obs)
