from AquaRL.mpi.BaseMPI import BaseMPI
from AquaRL.algo.PPO import PPO
from mpi4py import MPI
from AquaRL.worker.Worker import Worker
from AquaRL.policy.GaussianPolicy import GaussianPolicy
import numpy as np


class PPOMPI(BaseMPI):
    def __init__(self, hyper_parameters, actor: GaussianPolicy, critic, env, comm: MPI.COMM_WORLD, work_space: str,
                 env_args):
        super().__init__(comm, work_space, env_args)
        self.actor = actor

        if self.rank == 0:
            self.ppo = PPO(
                hyper_parameters=hyper_parameters,
                data_pool=self.main_data_pool,
                actor=actor,
                critic=critic,
                works_pace=work_space
            )

        else:
            self.worker = Worker(env, env_args, self.sub_data_pool, actor)

        # self.train()

    def train(self):
        sendbuf = np.zeros(100, dtype='i')
        recvbuf = None
        if self.rank == 0:
            recvbuf = np.empty([self.size, 100], dtype='i')

        for i in range(self.env_args.epochs):
            if self.rank > 0:
                std = np.empty(self.env_args.action_dims,dtype=np.float32)
                # send_wait = i
            else:
                self.ppo.actor.save_weights(self.cache_path)
                std = self.actor.get_std()
                print("main std:{},{}".format(i, self.actor.get_std()))
                # recv_wait = np.zeros((self.size - 1, 1), dtype=np.float32)

            self.comm.Bcast(std, root=0)
            # self.comm.Gather(sendbuf, recvbuf, root=0)
            self.comm.Barrier()

            if self.rank > 0:
                self.worker.policy.load_weights(self.cache_path)
                self.actor.set_std(std)
                # print("sub std:{},{}".format(i, self.actor.get_std()))
                self.worker.sample()
                # print(self.sub_data_pool.prob_buffer)

            # self.comm.Gather(sendbuf, recvbuf, root=0)
            self.comm.Barrier()

            if self.rank == 0:
                # self.main_data_pool.save_all_data(self.debug_path + '/' + 'main' + str(i))
                # print(self.main_data_pool.prob_buffer)
                self.ppo.optimize()
                # print(self.rank)
            # else:
            #     print(self.sub_data_pool.prob_buffer)
            # self.comm.Gather(sendbuf, recvbuf, root=0)
            self.comm.Barrier()

            # if self.rank == 0:
            #     self.main_data_pool.save_all_data(self.debug_path+'/'+'main'+str(i))
            # else:
            #     self.sub_data_pool.save_all_data(self.debug_path + '/'+'sub{}'.format(self.rank)+str(i))

    def close_shm(self):
        if self.rank == 0:
            self.main_data_pool.close_shm()
        else:
            self.sub_data_pool.close_shm()


