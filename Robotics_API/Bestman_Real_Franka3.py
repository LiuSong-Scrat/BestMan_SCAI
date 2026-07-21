import os 
import time
import sys
import signal
from franky import  *
from Config.utils import load_config
from .Pose import Pose
import numpy as np
import atexit
from scipy.spatial.transform import Rotation as R

class Bestman_Real_Franka3:
    SAFE_CARTESIAN_TRANSLATION_VEL = 0.40
    SAFE_CARTESIAN_ROTATION_VEL = 2.20
    SAFE_JOINT_VELOCITY_LIMITS = np.array([1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80])
    CARTESIAN_RELATIVE_DYNAMICS = (0.7, 0.5, 0.5)
    REACTION_RELATIVE_DYNAMICS = (0.15, 0.15, 0.10)

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
        self.robot = None
        self.gripper = None
        self._release_started = False
        self._shutdown_signal = None
        atexit.register(self.release_robot)
        pass

    def initialize_robot(self):
        #Ensure The Robot Unlocked and FCL is Open 
        try:
            self.robot_web_session.open()
            if not self.robot_web_session.has_control():
                try:
                    print("Start Try to Control the Robot")
                    self.robot_web_session.take_control(wait_timeout=3.0)
                except TakeControlTimeoutError:
                    print("Control is occupied by another session. Force takeover requires blue-button confirmation.")
                    self.robot_web_session.take_control(wait_timeout=30.0, force=True)
        except Exception as e:
            print(f"An error occurred: {e}")
            # 如果发生错误，建议确保释放控制权
            if self.robot_web_session.is_open:
                self.robot_web_session.close()
            exit(-1)
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

    def release_robot(self, stop_motion=True):
        if self._release_started:
            return
        self._release_started = True

        if stop_motion and self.robot is not None:
            try:
                self.robot.stop()
            except Exception as e:
                print(f"Stop robot motion failed during release: {e}")

        try:
            if self.robot_web_session.is_open:
                try:
                    if self.robot_web_session.has_control():
                        self.robot_web_session.release_control()
                except Exception as e:
                    print(f"Release control failed: {e}")
                self.robot_web_session.close()
                print("Control Released!")
        except Exception as e:
            print(f"Close robot web session failed: {e}")

    def install_exit_handlers(self):
        def _handle_exit_signal(signum, frame):
            self._shutdown_signal = signum
            signal_name = signal.Signals(signum).name
            print(f"Received {signal_name}; releasing robot control before exit.")
            self.release_robot()
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is not None:
                signal.signal(sig, _handle_exit_signal)


    def lock_robot(self):
        if self.robot_web_session.is_open:
            # Lock brakes
            self.robot_web_session.lock_brakes()
            # Disable the FCI
            self.robot_web_session.release_control()
            self.robot_web_session.close()
        return True
    
    def go_home(self,home_js = np.array([-0.0684262,-0.512061,0.0723129,-2.794,0.0469305,2.28233,0.749217])):
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
        reaction_motion = CartesianMotion(
            Affine([0.0, 0.0, -0.2]),
            ReferenceType.Relative,
            RelativeDynamicsFactor(*self.REACTION_RELATIVE_DYNAMICS),
        )  # Move up for 10cm
        # 如果检测到Z方向的力大于30牛顿，就触发定义好的reaction_motion
        reaction = Reaction(Measure.FORCE_Z < -50.0, reaction_motion) #向上的力大于3N
        motion.add_reaction(reaction)
        def reaction_callback(robot_state: RobotState, rel_time: float, abs_time: float):
            print(f"Reaction fired at {abs_time}.")
        reaction.register_callback(reaction_callback)

    def _set_cartesian_motion_limits(self, maxLinearVel, maxAngularVel):
        safe_linear_vel = min(float(maxLinearVel), self.SAFE_CARTESIAN_TRANSLATION_VEL)
        safe_angular_vel = min(float(maxAngularVel), self.SAFE_CARTESIAN_ROTATION_VEL)
        self.robot.translation_velocity_limit.set(safe_linear_vel)
        self.robot.rotation_velocity_limit.set(safe_angular_vel)
        self.robot.joint_velocity_limit.set(self.SAFE_JOINT_VELOCITY_LIMITS.copy())
        return safe_linear_vel, safe_angular_vel

    @staticmethod
    def _limit_vector_norm(vector, max_norm):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if max_norm > 0 and norm > max_norm:
            return vector * (max_norm / norm)
        return vector

    @staticmethod
    def _pose_to_affine(goal_pose):
        if not isinstance(goal_pose, Pose):
            raise TypeError("pose must be an instance of Pose")
        position = np.asarray(goal_pose.get_position(), dtype=float)
        orientation = np.asarray(goal_pose.get_orientation(type="quaternion"), dtype=float)
        return Affine(position, orientation)

    def _estimate_waypoint_velocities(
        self,
        goal_poses,
        waypoint_dt,
        maxLinearVel,
        maxAngularVel,
        stop_at_end,
        velocity_scale,
        min_position_step,
        min_rotation_step,
    ):
        waypoint_dt = max(float(waypoint_dt), 1e-3)
        positions = np.asarray([pose.get_position() for pose in goal_poses], dtype=float)
        rotations = R.from_quat([pose.get_orientation(type="quaternion") for pose in goal_poses])
        linear_velocities = []
        angular_velocities = []

        for idx in range(len(goal_poses)):
            if idx == 0 or idx == len(goal_poses) - 1:
                linear = np.zeros(3)
                angular = np.zeros(3)
            else:
                prev_step = positions[idx] - positions[idx - 1]
                next_step = positions[idx + 1] - positions[idx]
                prev_rotvec = (rotations[idx] * rotations[idx - 1].inv()).as_rotvec()
                next_rotvec = (rotations[idx + 1] * rotations[idx].inv()).as_rotvec()
                if min(np.linalg.norm(prev_step), np.linalg.norm(next_step)) < min_position_step:
                    linear = np.zeros(3)
                else:
                    linear = (positions[idx + 1] - positions[idx - 1]) / (2.0 * waypoint_dt)
                if min(np.linalg.norm(prev_rotvec), np.linalg.norm(next_rotvec)) < min_rotation_step:
                    angular = np.zeros(3)
                else:
                    delta_rot = rotations[idx + 1] * rotations[idx - 1].inv()
                    angular = delta_rot.as_rotvec() / (2.0 * waypoint_dt)

            linear_velocities.append(self._limit_vector_norm(linear * velocity_scale, maxLinearVel * velocity_scale))
            angular_velocities.append(self._limit_vector_norm(angular * velocity_scale, maxAngularVel * velocity_scale))

        if stop_at_end:
            linear_velocities[-1] = np.zeros(3)
            angular_velocities[-1] = np.zeros(3)

        return linear_velocities, angular_velocities

    def move_eef_to_goal_pose(self, goal_pose, maxLinearVel=0.3, maxAngularVel=1,asynchronous=False, settle_time=0.0):
        """
        Moves the end effector to the specified pose.
        Args:
            goal_pose (Pose): The desired pose (position and orientation in quaternion format).
            maxLinearVel (float): Maximum linear velocity. Defaults to 0.1 m/s.
            maxAngularVel (float): Maximum angular velocity. Defaults to 0.5 rad/s.
        Returns:
            None
        """
        self._set_cartesian_motion_limits(maxLinearVel, maxAngularVel)

        combined_transformation = self._pose_to_affine(goal_pose)
        execute_motion = CartesianMotion(
            combined_transformation,
            ReferenceType.Absolute,
            RelativeDynamicsFactor(*self.CARTESIAN_RELATIVE_DYNAMICS),
        )
        self.motion_move_safety_ensure(execute_motion)
        self.robot.move(execute_motion,asynchronous=asynchronous)
        if settle_time > 0:
            time.sleep(settle_time)

    def move_eef_through_goal_poses(
        self,
        goal_poses,
        maxLinearVel=0.3,
        maxAngularVel=1.0,
        waypoint_dt=0.08,
        asynchronous=False,
        stop_at_end=True,
        use_target_velocities=True,
        velocity_scale=0.35,
        min_position_step=0.003,
        min_rotation_step=0.02,
    ):
        """
        Move through a Cartesian pose chunk as one continuous waypoint motion.
        Intermediate waypoint velocities keep the arm from stopping at every policy action.
        """
        if len(goal_poses) == 0:
            return
        if len(goal_poses) == 1:
            self.move_eef_to_goal_pose(
                goal_poses[0],
                maxLinearVel=maxLinearVel,
                maxAngularVel=maxAngularVel,
                asynchronous=asynchronous,
            )
            return

        for goal_pose in goal_poses:
            if not isinstance(goal_pose, Pose):
                raise TypeError("goal_poses must contain Pose instances")

        safe_linear_vel, safe_angular_vel = self._set_cartesian_motion_limits(maxLinearVel, maxAngularVel)
        if use_target_velocities:
            linear_velocities, angular_velocities = self._estimate_waypoint_velocities(
                goal_poses,
                waypoint_dt,
                safe_linear_vel,
                safe_angular_vel,
                stop_at_end,
                velocity_scale,
                min_position_step,
                min_rotation_step,
            )

        waypoints = []
        for idx, goal_pose in enumerate(goal_poses):
            if use_target_velocities:
                state = CartesianState(
                    self._pose_to_affine(goal_pose),
                    Twist(
                        np.asarray(linear_velocities[idx], dtype=float),
                        np.asarray(angular_velocities[idx], dtype=float),
                    ),
                )
            else:
                state = CartesianState(self._pose_to_affine(goal_pose))
            waypoints.append(CartesianWaypoint(state))

        execute_motion = CartesianWaypointMotion(
            waypoints,
            relative_dynamics_factor=RelativeDynamicsFactor(*self.CARTESIAN_RELATIVE_DYNAMICS),
        )
        self.motion_move_safety_ensure(execute_motion)
        self.robot.move(execute_motion, asynchronous=asynchronous)


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
