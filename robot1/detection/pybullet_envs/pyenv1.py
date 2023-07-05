import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import cv2
import gym


class ArmEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\cube.urdf")
        self.arm_pos = [0, 0, 0.72]
        self.arm_q = [0, 0, 0, 1]
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.arm_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\new2.urdf", self.arm_pos, self.arm_q, useFixedBase=True, flags=flags)
        self.evn_pos = [0, 0, 1.3]
        self.evn_q = [0, 0, 0, 1]
        flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT | p.URDF_USE_INERTIA_FROM_FILE
        self.evn_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\evn.urdf", self.evn_pos, self.evn_q, useFixedBase=True, flags=flags)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.render_width = 640  # 渲染图像宽度
        self.render_height = 480  # 渲染图像高度
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + self.render_width * self.render_height,))
        # Enable collision detection between the arm and the environment
        p.setCollisionFilterPair(self.arm_id, self.evn_id, -1, -1, 1)
        self.camera_up_vector = [0, 0, 1]  # 假设Z轴为上方向
        self.fov = 60  # 视场角度
        self.aspect = self.render_width / self.render_height  # 宽高比
        self.near_plane = 0.1  # 近裁剪面
        self.far_plane = 100.0  # 远裁剪面
        self.end_effector_id = p.getNumJoints(self.arm_id) - 1  # 假设末端连杆的链接索引为最后一个连接索引

    def get_observation(self):
        # 获取机械臂状态和相机图像，构建观察空间
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
        view_matrix = p.computeViewMatrix(camera_pos, camera_pos + np.array([1, 0, 0]), self.camera_up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near_plane, self.far_plane)
        _, _, rgb_img, _, _ = p.getCameraImage(self.render_width, self.render_height, view_matrix, projection_matrix)
        rgb_img = np.reshape(rgb_img, (self.render_height, self.render_width, 4))[:, :, :3]  # 去除alpha通道
        rgb_img = np.flipud(rgb_img)  # 垂直翻转图像
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)  # 将RGB图像转换为灰度图像
        return gray_img

    def reset(self):
        # Reset the environment to a new episode
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(True)
        self.plane_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\cube.urdf")
        self.arm_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\new2.urdf", self.arm_pos, self.arm_q, useFixedBase=True)
        self.evn_id = p.loadURDF("C:\\Users\\User\\Documents\\drone detection\\1_robot1\\robot1\\detection\\pybullet_envs\\urdf\\evn.urdf", self.evn_pos, self.evn_q, useFixedBase=True)
        return self.get_observation()

    def step(self, action):
        # Execute an action and return the new observation, reward, and done flag
        joint_velocities = np.zeros(4)  # Assuming zero velocities for all joints
        joint_forces = np.ones(4) * 1000  # Assuming constant forces for all joints
        joint_positions = self.get_observation()
        target_position = self.calculate_target_position(action)
        joint_angles = self.calculate_inverse_kinematics(target_position)
        p.setJointMotorControlArray(
            bodyUniqueId=self.arm_id,
            jointIndices=[2, 3, 4, 5],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_angles[2:6],  # 提取与 jointIndices 相对应的关节角度
            targetVelocities=joint_velocities,
            forces=joint_forces,
            positionGains=[1.0] * 4
        )
        p.stepSimulation()
        target_positions = self.generate_target_positions()
        done = False
        for target_position in target_positions:
            joint_angles = self.calculate_inverse_kinematics(target_position)
            self.move_arm(joint_angles)
            observation = self.get_observation()
            reward = self.calculate_reward(observation, target_position)
            done = self.check_collision() or np.linalg.norm(observation[:3] - target_position) > 0.2
            if done:
                break
        observation = self.get_observation()
        reward = self.calculate_reward(observation, target_position)
        done = self.check_collision()
        return observation, reward, done, {}

    def generate_target_positions(self):
        range_min = np.array([-0.6, -0.2, 1.3])
        range_max = np.array([0.6, 0.2, 1.7])
        step_size = 0.01  # 步长设置为1cm
        x_values = np.arange(range_min[0], range_max[0] + step_size, step_size)
        y_values = np.arange(range_min[1], range_max[1] + step_size, step_size)
        z_values = np.arange(range_min[2], range_max[2] + step_size, step_size)
        target_positions = []
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    target_positions.append([x, y, z])
        return target_positions

    def calculate_inverse_kinematics(self, target_position):
        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.arm_id,
            endEffectorLinkIndex=self.end_effector_id,
            targetPosition=target_position,
            lowerLimits=[-np.pi / 2] * 4,
            upperLimits=[np.pi / 2] * 4,
            jointRanges=[np.pi] * 4,
            restPoses=self.get_arm_state(),
            maxNumIterations=100
        )
        return joint_angles

    def move_arm(self, joint_angles):
        joint_indices = [2, 3, 4, 5]  # Assuming the indices of the joints that control the arm
        for i in range(len(joint_indices)):
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=joint_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                positionGain=1.0
            )

    def calculate_target_position(self, action):
        # Convert the action values to target position
        x_range = [-0.6, 0.6]
        y_range = [-0.2, 0.2]
        z_range = [1.3, 1.7]
        x = np.interp(action[0], [-1, 1], x_range)
        y = np.interp(action[1], [-1, 1], y_range)
        z = np.interp(action[2], [-1, 1], z_range)
        return [x, y, z]

    def calculate_reward(self, observation, target_position):
        distance = np.linalg.norm(observation[:3] - target_position)
        reward = -distance
        return reward

    def check_collision(self):
        contact_points = p.getContactPoints(self.arm_id, self.evn_id)
        return len(contact_points) > 0
    
env = ArmEnv()
observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print("Observation:", observation)
    print("Reward:", reward)