from mpi4py import MPI
import os
import tensorflow as tf
from AquaRL.pool.SharedPool import MainThreadSharaMemery, SubThreadShareMemery
from AquaRL.args import EnvArgs
import time


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


class BaseMPI:
    def __init__(self, comm: MPI.COMM_WORLD, work_space: str, env_args: EnvArgs, ALL_GPU=False):
        self.comm = comm
        # serial number
        self.rank = self.comm.Get_rank()
        # print(self.rank)
        self.env_args = env_args
        self.size = self.comm.Get_size()

        self.log_path = work_space + '/log'
        self.history_model_path = work_space + '/model'
        self.cache_path = work_space + '/cache'
        self.work_space = work_space
        self.debug_path = work_space + '/debug'

        start_step_pointer = (self.rank - 1) * self.env_args.thread_step
        start_epoch_pointer = (self.rank - 1) * self.env_args.trajs

        if self.rank == 0:
            mkdir(work_space)
            mkdir(self.history_model_path)
            mkdir(self.cache_path)
            mkdir(self.debug_path)

            self.main_data_pool = MainThreadSharaMemery(env_args.observation_dims, env_args.action_dims,
                                                        env_args.total_steps, env_args.epochs, env_args.total_trajs,
                                                        work_space)

            self.main_data_pool.pool_info()

        else:
            time.sleep(8)
            self.sub_data_pool = SubThreadShareMemery(env_args.observation_dims, env_args.action_dims,
                                                      env_args.total_steps, start_step_pointer, env_args.thread_step,
                                                      env_args.epochs, start_epoch_pointer, env_args.total_trajs,
                                                      work_space)

        self.comm.Barrier()

    # def sample(self):
    #     if self.rank == 0:
    #         self.worker.sample()

    @staticmethod
    def initial_GPU(rank):
        if rank == 0:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                for k in range(len(physical_devices)):
                    tf.config.experimental.set_memory_growth(physical_devices[k], True)
                    print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
            else:
                print("Not enough GPU hardware devices available")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    def close_shm(self):
        if self.rank == 0:
            self.main_data_pool.close_shm()
        else:
            self.sub_data_pool.close_shm()

