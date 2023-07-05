import sys
sys.path.append('..')
from detection.pybullet_envs.pyenv import ArmEnv
from detection.ppo.ppo import ppo
from spinup.utils.mpi_tools import mpi_fork
import detection.ppo.core as core

'''
这部分定义了命令行参数解析器，用于从命令行中读取参数。例如，--is-render和--is-good-view是用来控制是否显示渲染窗口和是否使用好的视角。
--hid是隐藏层的大小，--l是隐藏层的层数，--gamma是折扣因子，--seed是随机种子，--cpu是CPU数量，--epochs是训练轮数，
--exp_name是实验名称，--log_dir是日志存储路径。
'''
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--is-render',action="store_true")
    parser.add_argument('--is-good-view',action="store_true")

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='pyevn')
    parser.add_argument('--log_dir', type=str, default="./logs")
    args = parser.parse_args()

    env=ArmEnv()
    
    #通过mpi_fork函数以MPI方式运行并行代码。
    mpi_fork(args.cpu)  # run parallel code with mpi
    #导入setup_logger_kwargs函数，并使用实验名称和随机种子等参数设置logger_kwargs，用于配置日志记录。
    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir=args.log_dir)

    '''
    调用ppo函数来运行PPO算法。传入的参数包括环境实例env，actor_critic指定了使用的策略和值函数网络模型，
    ac_kwargs是用于构建网络的参数字典，gamma是折扣因子，seed是随机种子，steps_per_epoch是每个轮数的步数，
    epochs是训练轮数，logger_kwargs是用于配置日志记录的参数。
    '''
    ppo(env,
        actor_critic=core.CNNActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=100*args.cpu,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)
