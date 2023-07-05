import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import cv2
import gym
import time


class ArmEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")  # Replace with the path to your plane URDF file
        self.arm_pos = [0, 0, 0.72]
        self.arm_q = [0, 0, 0, 1]
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.arm_id = p.loadURDF("robot.urdf", self.arm_pos, self.arm_q, useFixedBase=True, flags=flags)  # Replace with the path to your robot URDF file
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.render_width = 640  # Rendered image width
        self.render_height = 480  # Rendered image height
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + self.render_width * self.render_height,))
        self.end_effector_id = p.getNumJoints(self.arm_id) - 1  # Assuming the end effector link index is the last link

        # Object detection variables
        self.object_id = None
        self.object_detected = False

    def render(self):
        # Set up camera parameters
        camera_distance = 2.5
        camera_yaw = 45
        camera_pitch = -30
        target_position = [0, 0, 0]  # Position to look at

        # Reset the camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_position,
        )

    def get_observation(self):
        # Get the arm state and camera image, and construct the observation space
        arm_state = self.get_arm_state()
        gray_img = self.get_camera_image()
        return np.concatenate((arm_state, gray_img.flatten()))

    def get_arm_state(self):
        joint_states = p.getJointStates(self.arm_id, range(p.getNumJoints(self.arm_id)))
        joint_positions = [state[0] for state in joint_states]
        return np.array(joint_positions)

    def get_camera_image(self):
        end_effector_info = p.getLinkState(self.arm_id, self.end_effector_id)
        camera_pos = end_effector_info[0]
        camera_orientation = end_effector_info[1]
        view_matrix = p.computeViewMatrix(camera_pos, camera_pos + np.array([1, 0, 0]), [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(60, self.render_width / self.render_height, 0.1, 100.0)
        _, _, rgb_img, _, _ = p.getCameraImage(self.render_width, self.render_height, view_matrix, projection_matrix)
        rgb_img = np.reshape(rgb_img, (self.render_height, self.render_width, 4))[:, :, :3]  # Remove alpha channel
        rgb_img = np.flipud(rgb_img)  # Flip the image vertically
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)  # Convert RGB image to grayscale
        return gray_img

    def calculate_inverse_kinematics(self, target_pos):
        # Replace this function with your own inverse kinematics calculation specific to your robot arm
        return target_pos

    def reset(self):
        # Reset the robot arm to the initial position
        p.resetBasePositionAndOrientation(self.arm_id, self.arm_pos, self.arm_q)
        p.stepSimulation()
        time.sleep(0.1)

        # Load and place the object in the environment
        self.object_id = p.loadURDF("object.urdf", [0.5, 0, 0.75])  # Replace with the path to your object URDF file

        # Reset object detection flag
        self.object_detected = False

        return self.get_observation()

    def step(self, action):
        # Scale action values to joint limits
        scaled_action = np.array(action) * np.pi / 2

        # Apply action to robot joints
        joint_positions = self.get_arm_state()
        joint_positions += scaled_action
        p.setJointMotorControlArray(
            self.arm_id,
            range(p.getNumJoints(self.arm_id)),
            p.POSITION_CONTROL,
            targetPositions=joint_positions.tolist(),
        )
        p.stepSimulation()
        time.sleep(0.01)

        # Check for object detection
        contact_points = p.getContactPoints(self.arm_id, self.object_id)
        if len(contact_points) > 0:
            self.object_detected = True

        # Compute reward
        reward = 1.0 if self.object_detected else 0.0

        # Check for episode termination
        done = self.object_detected

        # Get the current observation
        observation = self.get_observation()

        return observation, reward, done, {}

    def close(self):
        p.disconnect()


env = ArmEnv()
observation = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Replace with your own action selection logic
    observation, reward, done, _ = env.step(action)
    if env.object_detected:
        print("Object detected!")
