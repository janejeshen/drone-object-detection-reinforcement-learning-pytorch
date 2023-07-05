import sys
sys.path.append('..')
from detection.pybullet_envs.pyenv import ArmEnv
from detection.ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import detection.ppo_gpu.core_gpu as core

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--is-render',action="store_true")
    parser.add_argument('--is-good-view',action="store_true")

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='pyevn')
    parser.add_argument('--log_dir', type=str, default="./logs")
    args = parser.parse_args()

    env=ArmEnv()
    

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(env,
        actor=core.CNNGaussianActor,
        critic=core.CNNCritic,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=env.kMaxEpisodeSteps*args.cpu,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)
