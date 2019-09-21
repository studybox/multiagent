import tensorflow as tf
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from acktr_graph import learn
#from baselines.acktr.policies import GaussianMlpPolicy
#from baselines.acktr.value_functions import NeuralNetValueFunction
from policies import CategoricalGraphPolicy
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
import os
from rl import bench
from MCE_env import MceEnv
import logging
import gym


def create_env(seed, rank):
    def _thunk():
            env = MceEnv()
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
    return _thunk

def train(num_timesteps, seed):
    set_global_seeds(seed)
    num_cpu = 32
    env = SubprocVecEnv([create_env(seed, i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = CategoricalGraphPolicy # TODO
    #vf = NeuralNetValueFunction
    learn(policy_fn, None,  env, seed = 1, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)
def main():
    logdir = './log_graph'
    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    seed = 1
    train(150000, seed)
if __name__ == "__main__":
    main()
