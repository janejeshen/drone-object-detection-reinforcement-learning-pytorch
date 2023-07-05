import sys
sys.path.append('..')
from detection.pybullet_envs.pyenv import ArmEnv
import torch

env = ArmEnv()
obs = env.reset()

ac = torch.load("../saved_models/model.pt")

actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))

sum_reward = 0
success_times = 0
for i in range(50):
    obs = env.reset()
    for step in range(1000):
        actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, info = env.step(actions)
        if reward == 1:
            success_times += 1
        if done:
            break

print('sum reward={}'.format(sum_reward))
print('success rate={}'.format(success_times / 50))



