from AquaRL.mpi.BaseMPI import BaseMPI
from AquaRL.algo.GAIL import GAIL
from mpi4py import MPI
from AquaRL.worker.Worker import Worker
from AquaRL.args import GAILParameters
from AquaRL.args import PPOHyperParameters
from AquaRL.algo.PPO import PPO
import numpy as np


class GAILPPO(BaseMPI):
    def __init__(self, gail_parameters: GAILParameters, ppo_parameters: PPOHyperParameters,
                 discriminator, actor,
                 critic, env, expert_data_pool, comm: MPI.COMM_WORLD, work_space: str, env_args, action_fun=None):
        super().__init__(comm, work_space, env_args)
        self.actor = actor
        self.action_fun = action_fun

        if self.rank == 0:
            self.ppo = PPO(
                hyper_parameters=ppo_parameters,
                data_pool=self.main_data_pool,
                actor=actor,
                critic=critic,
                works_pace=work_space,
                discriminator=discriminator
            )

            self.gail = GAIL(
                parameters=gail_parameters,
                expert_data_pool=expert_data_pool,
                discriminator=discriminator,
                data_pool=self.main_data_pool,
                work_space=work_space
            )

        else:
            self.worker = Worker(env, env_args, self.sub_data_pool, actor, action_fun)

        self.comm.Barrier()

    def train(self):
        for i in range(self.env_args.epochs):
            if self.rank > 0:
                std = np.empty(self.env_args.action_dims, dtype=np.float32)
            else:
                self.ppo.actor.save_weights(self.cache_path)
                std = self.actor.get_std()
                print("main std:{},{}".format(i, self.actor.get_std()))

            self.comm.Bcast(std, root=0)
            self.comm.Barrier()
            if self.rank > 0:
                self.worker.policy.load_weights(self.cache_path)
                self.actor.set_std(std)
                self.worker.sample()

            self.comm.Barrier()

            if self.rank == 0:
                self.gail.optimize()
                self.ppo.optimize()

            self.comm.Barrier()
