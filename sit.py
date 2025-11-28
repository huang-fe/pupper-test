import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

np.set_printoptions(precision=3, suppress=True)


# -----------------------------
# Transform helpers
# -----------------------------

def rotation_x(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ])

def rotation_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1],
    ])

def rotation_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

def translation(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


# =============================
#      INVERSE KINEMATICS
# =============================

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('sit_stand_kinematics')

        # Sub & Pub
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_cb,
            10
        )
        self.command_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_command_controller/commands',
            10
        )

        self.joint_positions = None
        self.joint_velocities = None

        # modes: "stand", "sit", "idle"
        self.mode = "stand"

        # These will be filled by trajectory builder
        self.motion_cache = None
        self.motion_index = 0

        # Sitting & standing default foot locations
        self.default_stand_ee = {
            0: np.array([0.06, -0.09, -0.14]),   # FR
            1: np.array([0.06,  0.09, -0.14]),   # FL
            2: np.array([-0.11,-0.09, -0.14]),   # BR
            3: np.array([-0.11, 0.09, -0.14])    # BL
        }

        self.default_sit_ee = {
            0: np.array([0.06, -0.09, -0.14]),   # front legs unchanged
            1: np.array([0.06,  0.09, -0.14]),
            2: np.array([-0.03, -0.09, -0.05]),  # back legs folded under
            3: np.array([-0.03,  0.09, -0.05])
        }

        # FK for each leg index
        self.fk_functions = [
            self.fr_leg_fk, self.fl_leg_fk,
            self.br_leg_fk, self.bl_leg_fk
        ]

        # Timers
        self.ik_timer = self.create_timer(1/100.0, self.ik_cb)
        self.pd_timer = self.create_timer(1/200.0, self.pd_cb)

    # -----------------------------
    # Forward Kinematics for each leg
    # -----------------------------
    def fr_leg_fk(self, theta):
        T1 = translation(0.075, -0.0835, 0) @ rotation_x(1.5708) @ rotation_z(theta[0])
        T2 = rotation_y(-1.5708) @ rotation_z(theta[1])
        T3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
        T4 = translation(0.06231, -0.06216, 0.018)
        return (T1 @ T2 @ T3 @ T4)[:3, 3]

    def fl_leg_fk(self, theta):
        T1 = translation(0.075, -0.0445, 0) @ rotation_x(1.5708) @ rotation_z(theta[0])
        T2 = translation(0,0,0.039) @ rotation_y(-1.5708) @ rotation_z(theta[1])
        T3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
        T4 = translation(0.06231, -0.06216, 0.018)
        return (T1 @ T2 @ T3 @ T4)[:3, 3]

    def br_leg_fk(self, theta):
        T1 = translation(-0.075, -0.0835, 0) @ rotation_x(1.5708) @ rotation_z(theta[0])
        T2 = rotation_y(-1.5708) @ rotation_z(theta[1])
        T3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
        T4 = translation(0.06231, -0.06216, 0.018)
        return (T1 @ T2 @ T3 @ T4)[:3, 3]

    def bl_leg_fk(self, theta):
        T1 = translation(-0.075, 0.0445, 0) @ rotation_x(-1.5708) @ rotation_z(theta[0])
        T2 = translation(0,0,0.039) @ rotation_y(-1.5708) @ rotation_z(theta[1])
        T3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
        T4 = translation(0.06231, -0.06216, 0.018)
        return (T1 @ T2 @ T3 @ T4)[:3,3]

    def forward_kinematics(self, theta):
        return np.concatenate([self.fk_functions[i](theta[3*i: 3*i+3]) for i in range(4)])

    # -----------------------------
    # Inverse Kinematics (gradient descent)
    # -----------------------------
    def inverse_kinematics_single_leg(self, target_ee, leg_index, initial=[0,0,0]):
        fk = self.fk_functions[leg_index]

        def cost(theta):
            return np.linalg.norm(fk(theta) - target_ee)**2

        def grad(theta, eps=1e-3):
            g = np.zeros(3)
            for i in range(3):
                t_plus = theta.copy();  t_plus[i] += eps
                t_minus = theta.copy(); t_minus[i] -= eps
                g[i] = (cost(t_plus) - cost(t_minus)) / (2*eps)
            return g

        theta = np.array(initial)
        lr = 8
        for _ in range(25):
            theta -= lr * grad(theta)
        return theta

    # -----------------------------
    # Trajectory generator
    # -----------------------------
    def generate_motion(self, target_ee_dict, duration=1.5, dt=0.02):
        steps = int(duration/dt)

        # Current EE positions
        current = self.forward_kinematics(self.joint_positions).reshape(4,3)

        trajectory = []

        for t in range(steps):
            s = t / (steps-1)
            interp_targets = []

            # interpolate each foot
            for leg in range(4):
                start = current[leg]
                end = target_ee_dict[leg]
                interp = (1-s)*start + s*end
                interp_targets.append(interp)

            # IK for all legs
            joint_list = []
            for leg in range(4):
                ee = interp_targets[leg]
                j = self.inverse_kinematics_single_leg(ee, leg)
                joint_list.append(j)

            trajectory.append(np.concatenate(joint_list))

        return np.array(trajectory)

    # -----------------------------
    # ROS Callbacks
    # -----------------------------
    def joint_state_cb(self, msg):
        names = [
            'leg_front_r_1','leg_front_r_2','leg_front_r_3',
            'leg_front_l_1','leg_front_l_2','leg_front_l_3',
            'leg_back_r_1','leg_back_r_2','leg_back_r_3',
            'leg_back_l_1','leg_back_l_2','leg_back_l_3'
        ]
        self.joint_positions = np.array([msg.position[msg.name.index(n)] for n in names])
        self.joint_velocities = np.array([msg.velocity[msg.name.index(n)] for n in names])

    def ik_cb(self):
        if self.joint_positions is None:
            return

        # Build trajectory if needed
        if self.motion_cache is None:
            if self.mode == "sit":
                self.motion_cache = self.generate_motion(self.default_sit_ee)
                self.motion_index = 0
            elif self.mode == "stand":
                self.motion_cache = self.generate_motion(self.default_stand_ee)
                self.motion_index = 0
            else:
                return

        # Follow trajectory
        if self.motion_index < len(self.motion_cache):
            self.target_joint_positions = self.motion_cache[self.motion_index]
            self.motion_index += 1
        else:
            # Hold last frame
            self.target_joint_positions = self.motion_cache[-1]

    def pd_cb(self):
        if hasattr(self, "target_joint_positions"):
            msg = Float64MultiArray()
            msg.data = self.target_joint_positions.tolist()
            self.command_pub.publish(msg)

def main():
    rclpy.init()
    node = InverseKinematics()

    try:
        print("\nCommands:")
        print("  node.mode='sit'   → robot sits")
        print("  node.mode='stand' → robot stands up\n")

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        zero = Float64MultiArray()
        zero.data = [0.0]*12
        node.command_pub.publish(zero)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
