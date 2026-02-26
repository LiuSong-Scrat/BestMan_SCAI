import os 
import time
import sys
import signal
from franky import  *
from Config.utils import load_config
from .Pose import Pose
import numpy as np
import atexit

class Bestman_Real_Franka3:
    def __init__(self):

        # load Configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "../Config/default_franka3.yaml")
        self.cfg = load_config(config_file)
        self.robot_web_session = RobotWebSession(
            self.cfg.Robot.fci_ip, 
            self.cfg.Robot.username, 
            self.cfg.Robot.password
        )
        pass

    def initialize_robot(self):
        #Ensure The Robot Unlocked and FCL is Open 
        try:
            self.robot_web_session.open()
            try:
                print("Start Try to Control the Robot")
                self.robot_web_session.take_control(wait_timeout=0.1)
            except TakeControlTimeoutError:
                print("Start Force to Control the Robot, Please Press the button")
                self.robot_web_session.take_control(wait_timeout=30.0, force=True)
        except Exception as e:
            print(f"An error occurred: {e}")
            # 如果发生错误，建议确保释放控制权
            if self.robot_web_session.is_open:
                self.robot_web_session.close()
            exit(-1)
        atexit.register(self.release_robot)
        # Unlock the brakes
        self.robot_web_session.unlock_brakes()
        # Enable the FCI
        self.robot_web_session.enable_fci()
        time.sleep(3) #wait for mode initialize


        # Create The Robot and Gripper Object
        self.robot = Robot(self.cfg.Robot.fci_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(self.cfg.Robot.velocity,self.cfg.Robot.acceleration,self.cfg.Robot.jerk)
        self.gripper = Gripper(self.cfg.Robot.fci_ip)

        self.initial_robot_states = self.robot.current_cartesian_state

        self.robot.set_cartesian_impedance(
            [10, 10, 10, 2, 2, 2]
        )

        return True
    def release_robot(self):
        if self.robot_web_session.is_open:
            self.robot_web_session.close()
            print("Control Released!")


    def lock_robot(self):
        if self.robot_web_session.is_open:
            self.robot_web_session.close()

            # Lock brakes
            self.robot_web_session.lock_brakes()
            # Disable the FCI
            self.robot_web_session.release_control()
        return True
    
    def go_home(self,home_js = np.array([-0.068755,-0.511863,0.072686,-2.79413,0.0465087,2.28272,-2.39184])):
        start_qpos = JointMotion(home_js)
        self.robot.move(start_qpos)

    def open_gripper(self, speed=0.1, force=0):
        # 打开夹爪，释放物体
        self.gripper.open(speed)

    def open_gripper_width(self,width=0, speed=0.1, force=5):
        self.gripper.move(width, speed)

    def close_gripper(self, speed=0.02, force=5):
        self.gripper.grasp(0.0, speed, force, epsilon_outer=1.0)

    def motion_move_safety_ensure(self,motion):
        # 创建一个反应动作，如果某种事件发生了，机器人就会执行该动作。
        reaction_motion = CartesianMotion(Affine([0.0, 0.0, -0.2]), ReferenceType.Relative)  # Move up for 10cm
        # 如果检测到Z方向的力大于30牛顿，就触发定义好的reaction_motion
        reaction = Reaction(Measure.FORCE_Z < -50.0, reaction_motion) #向上的力大于3N
        motion.add_reaction(reaction)
        def reaction_callback(robot_state: RobotState, rel_time: float, abs_time: float):
            print(f"Reaction fired at {abs_time}.")
        reaction.register_callback(reaction_callback)

    def move_eef_to_goal_pose(self, goal_pose, maxLinearVel=0.3, maxAngularVel=1,asynchronous=False):
        """
        Moves the end effector to the specified pose.
        Args:
            goal_pose (Pose): The desired pose (position and orientation in quaternion format).
            maxLinearVel (float): Maximum linear velocity. Defaults to 0.1 m/s.
            maxAngularVel (float): Maximum angular velocity. Defaults to 0.5 rad/s.
        Returns:
            None
        """
        self.robot.translation_velocity_limit.set(maxLinearVel)
        self.robot.rotation_velocity_limit.set(maxLinearVel)
        self.robot.joint_velocity_limit.set(np.array([maxAngularVel]).repeat(7))

        if not isinstance(goal_pose, Pose):
            raise TypeError("pose must be an instance of Pose")

        position = goal_pose.get_position()
        orientation = goal_pose.get_orientation(type="quaternion")  # R library return (qx,qy,qz,qw)
        combined_transformation = Affine(position,orientation)       # flexiv require (qw,qx,qy,qz)
        execute_motion = CartesianMotion(combined_transformation)
        self.motion_move_safety_ensure(execute_motion)
        self.robot.move(execute_motion,asynchronous=asynchronous)
        
        time.sleep(0.15)


    def get_current_eef_pose(self):
        current_robot_states = self.robot.current_cartesian_state
        current_position = current_robot_states.pose.end_effector_pose.translation
        current_orientation = current_robot_states.pose.end_effector_pose.quaternion
        current_pose = Pose(current_position, current_orientation)

        return current_pose
    def get_current_joint_values(self):
        cur_jpos = self.robot.current_joint_state.position
        return cur_jpos
   
    def move_arm_to_joint_values(self, joint_values, target_vel=None, target_acc=None, MAX_VEL=1, MAX_ACC=0.5):
        self.robot.joint_velocity_limit.set(np.array([MAX_VEL]).repeat(7))
        self.robot.joint_acceleration_limit.set(np.array([MAX_ACC]).repeat(7))
        joint_values = JointState(joint_values)
        motion_to_target_qpos = JointMotion(joint_values)
        self.robot.move(motion_to_target_qpos)