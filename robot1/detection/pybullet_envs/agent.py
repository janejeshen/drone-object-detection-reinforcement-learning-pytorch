import gym
import pybullet as p
import time
from policy_networkk import RobotArmEnv

RobotArmEnv()
# Create the simulated environment
env_name = "RobotArmEnv-v0"
env = gym.make(env_name)

# Set up the PyBullet physics simulation
p.connect(p.GUI)
p.setRealTimeSimulation(1)

# Reset the environment
state = env.reset()

# Run the simulation and visualize the result
while True:
    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_space.shape)
    
    next_state, reward, done, _ = env.step(action)
    
    # Visualize the robot arm and the drone in the simulation
    # You can customize the visualization based on your specific environment and objects
    
    p.stepSimulation()
    time.sleep(0.01)
    
    if done:
        break

# Close the PyBullet simulation
p.disconnect()