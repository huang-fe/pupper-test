import time
import numpy as np

class QuadrupedController:
    def __init__(self, robot_interface):
        self.robot = robot_interface

        # Joint name → index mapping
        self.names = [
            'leg_front_r_1','leg_front_r_2','leg_front_r_3',
            'leg_front_l_1','leg_front_l_2','leg_front_l_3',
            'leg_back_r_1','leg_back_r_2','leg_back_r_3',
            'leg_back_l_1','leg_back_l_2','leg_back_l_3'
        ]

        # Identify ONLY the joints we care about
        self.back_r_hip   = self.names.index('leg_back_r_1')
        self.back_r_ankle = self.names.index('leg_back_r_3')
        self.back_l_hip   = self.names.index('leg_back_l_1')
        self.back_l_ankle = self.names.index('leg_back_l_3')

    # ---- Utility function for smooth motion ----
    def interpolate_and_send(self, start, target, duration=2.0, steps=200):
        """
        Slowly interpolates between joint angles.
        duration = longer → slower motion
        """
        for i in range(steps):
            alpha = i / (steps - 1)
            cmd = (1 - alpha) * start + alpha * target
            self.robot.set_joint_positions(cmd)
            time.sleep(duration / steps)

    # ==========================
    #        SIT DOWN
    # ==========================
    def sit(self, duration=3.0):
        current = np.array(self.robot.get_joint_positions())

        # Desired sitting posture:
        # hips rotate backward (positive)
        # ankles fold forward (negative)
        target = current.copy()

        sit_hip_angle   = np.radians(10)   # backward hip bend
        sit_ankle_angle = np.radians(-10)  # ankle folding

        target[self.back_r_hip]   = sit_hip_angle
        target[self.back_l_hip]   = sit_hip_angle
        target[self.back_r_ankle] = sit_ankle_angle
        target[self.back_l_ankle] = sit_ankle_angle

        # Smooth transition
        self.interpolate_and_send(current, target, duration=duration, steps=300)

    # ==========================
    #       STAND BACK UP
    # ==========================
    def stand(self, duration=3.0):
        current = np.array(self.robot.get_joint_positions())
        target = current.copy()

        # Standing = everything back to neutral 0
        target[self.back_r_hip]   = 0.0
        target[self.back_l_hip]   = 0.0
        target[self.back_r_ankle] = 0.0
        target[self.back_l_ankle] = 0.0

        self.interpolate_and_send(current, target, duration=duration, steps=300)

