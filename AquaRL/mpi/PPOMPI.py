from AquaRL.mpi.BaseMPI import BaseMPI
from AquaRL.algo.PPO import PPO
from mpi4py import MPI
from AquaRL.worker.Worker import Worker
import numpy as np


class PPOMPI(BaseMPI):
    def __init__(self, hyper_parameters, env, comm: MPI.COMM_WORLD, work_space: str,
                 env_args, actor=None, critic=None, actor_critic=None, action_fun=None):
        super().__init__(comm, work_space, env_args)
        self.hyper_parameters = hyper_parameters
        if actor_critic is not None:
            self.actor = actor_critic
        else:
            self.actor = actor

        if self.rank == 0:
            self.ppo = PPO(
                hyper_parameters=hyper_parameters,
                data_pool=self.main_data_pool,
                actor=actor,
                critic=critic,
                actor_critic=actor_critic,
                works_pace=work_space
            )

        else:
            self.worker = Worker(
                env=env,
                env_args=env_args,
                data_pool=self.sub_data_pool,
                policy=self.actor,
                is_training=True,
                action_fun=action_fun
            )

        # self.train()

    def train(self):
        for i in range(self.env_args.epochs):
            if self.rank > 0:
                std = np.empty(self.env_args.action_dims, dtype=np.float32)
            else:
                self.actor.save_weights(self.cache_path)
                std = self.actor.get_std()
            self.comm.Bcast(std, root=0)
            self.comm.Barrier()

            if self.rank > 0:
                self.worker.policy.load_weights(self.cache_path)
                self.actor.set_std(std)
                if self.env_args.train_rnn_r2d2:
                    self.worker.sample_rnn()
                else:
                    self.worker.sample()
            self.comm.Barrier()

            if self.rank == 0:
                self.ppo.optimize()
            self.comm.Barrier()
