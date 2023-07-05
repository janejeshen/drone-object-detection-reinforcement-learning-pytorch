import sys
sys.path.append('..')
from detection.pybullet_envs.pyenv import ArmEnv

import numpy as np

'''
定义了一个名为test_arm_env的测试函数。在函数内部，创建了ArmEnv环境的实例env，然后调用env.reset()初始化环境并获取初始观察值observation。
断言observation.shape == (4 + env.render_width * env.render_height * 3,)用于检查观察值的形状是否与预期一致。

接下来，从动作空间中随机采样一个动作action，然后调用env.step(action)执行该动作，获取新的观察值observation、奖励reward、完成标志done和额外信息_。
再次使用断言observation.shape == (4 + env.render_width * env.render_height * 3,)来检查新的观察值的形状是否与预期一致。
'''
def test_arm_env():
    env = ArmEnv()
    observation = env.reset()
    assert observation.shape == (4 + env.render_width * env.render_height * 3,)
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    assert observation.shape == (4 + env.render_width * env.render_height * 3,)

'''
这部分代码用于在脚本直接执行时运行测试函数test_arm_env()。当脚本作为主程序执行时，__name__将被设置为'__main__'，因此测试函数将被调用。
'''
if __name__ == '__main__':
    test_arm_env()