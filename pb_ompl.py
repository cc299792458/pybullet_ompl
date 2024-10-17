import sys
import pybullet as p
import pybullet_data
import os.path as osp
from itertools import product
import time
import copy

# Import OMPL (assumes that OMPL is properly installed)
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    # If the OMPL module is not in the PYTHONPATH, assume it is installed in a subdirectory
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc

# Custom utilities - these should be part of your codebase
import utils  # Ensure you have a `utils.py` with necessary functions like collision checks, etc.

# Constants for the demo
INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 0.5


class PbOMPLRobot:
    '''
    Class representing the robot and its joint states for planning.
    '''
    def __init__(self, robot_id) -> None:
        self.id = robot_id

        # Prune fixed joints
        all_joint_num = p.getNumJoints(robot_id)
        all_joint_idx = list(range(all_joint_num))
        self.joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(self.joint_idx)
        self.joint_bounds = []
        self.state = None

        print(self.joint_idx)
        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds from PyBullet.
        '''
        for joint_id in self.joint_idx:
            joint_info = p.getJointInfo(self.id, joint_id)
            low, high = joint_info[8], joint_info[9]  # Joint limits
            if low < high:
                self.joint_bounds.append([low, high])
        print(f"Joint bounds: {self.joint_bounds}")
        return self.joint_bounds

    def get_cur_state(self):
        return self.state.copy()

    def set_state(self, state):
        '''
        Set the robot state (angles and velocities) for collision checking.
        '''
        self._set_joint_positions(self.joint_idx, state[:self.num_dim])
        self.state = state

    def reset(self):
        '''
        Reset the robot state to default.
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)


class PbStateSpace(ob.RealVectorStateSpace):
    '''
    Custom state space that includes joint angles and velocities.
    '''
    def __init__(self, num_dim) -> None:
        super().__init__(2 * num_dim)  # Include both angles and velocities
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        Allocate a state sampler for OMPL planning.
        '''
        return self.state_sampler if self.state_sampler else self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Set a custom state sampler.
        '''
        self.state_sampler = state_sampler


class PbOMPL:
    '''
    Class to handle OMPL planning, including setup for state space, control space, and collision checking.
    '''
    def __init__(self, robot, obstacles=[]) -> None:
        '''
        Initialize PbOMPL object with the robot and obstacles.
        '''
        self.robot = robot
        self.obstacles = obstacles

        # Create state space for joint angles and velocities (2 * num_dim)
        self.space = PbStateSpace(robot.num_dim)

        # Create control space for joint torques (or accelerations)
        self.control_space = oc.RealVectorControlSpace(self.space, robot.num_dim)

        # Set up space information for kinodynamic planning
        self.si = oc.SpaceInformation(self.space, self.control_space)
        self.ss = oc.SimpleSetup(self.si)

        # Set joint and velocity bounds in the state space
        self.set_joint_and_velocity_bounds()

        # Set a validity checker for collision checking
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        # Set control bounds
        self.set_control_bounds()

        # Set obstacles for the environment
        self.set_obstacles(obstacles)

        # Define the state propagator
        self.set_state_propagator()

        # Initialize the planner (e.g., RRT, KPIECE1)
        self.set_planner("RRT")  # Default to RRT

    def set_joint_and_velocity_bounds(self):
        '''Set the joint position and velocity bounds in the state space.'''
        bounds = ob.RealVectorBounds(2 * self.robot.num_dim)

        # Get joint position bounds
        joint_bounds = self.robot.get_joint_bounds()

        # Set bounds for joint angles
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])

        # Set bounds for joint velocities
        velocity_limits = 5.0  # Example: customize velocity limits
        for i in range(self.robot.num_dim):
            bounds.setLow(self.robot.num_dim + i, -velocity_limits)
            bounds.setHigh(self.robot.num_dim + i, velocity_limits)

        self.space.setBounds(bounds)

    def set_control_bounds(self):
        '''Set the control bounds in the control space (e.g., joint torques or accelerations).'''
        bounds = ob.RealVectorBounds(self.robot.num_dim)
        torque_limits = 10.0  # Customize torque limits based on your robot
        for i in range(self.robot.num_dim):
            bounds.setLow(i, -torque_limits)
            bounds.setHigh(i, torque_limits)
        self.control_space.setBounds(bounds)

    def set_obstacles(self, obstacles):
        '''Set obstacles in the environment and set up collision checking.'''
        self.obstacles = obstacles
        self.setup_collision_detection(self.robot, self.obstacles)

    def is_state_valid(self, state):
        '''Check whether a state is valid (i.e., collision-free).'''
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot.id, link1, self.robot.id, link2):
                return False
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions=True, allow_collision_links=[]):
        '''Set up self-collision and environment collision checking.'''
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            item for item in utils.get_moving_links(robot.id, robot.joint_idx) if item not in allow_collision_links)
        self.check_body_pairs = list(product([(robot.id, moving_links)], obstacles))

    def set_state_propagator(self):
        '''Set the state propagator to propagate the state using the system's dynamics.'''
        def propagate(state, control, duration, result):
            # Example: simple propagation assuming velocity control
            for i in range(self.robot.num_dim):
                result[i] = state[i] + state[self.robot.num_dim + i] * duration  # Update position
                result[self.robot.num_dim + i] = state[self.robot.num_dim + i] + control[i] * duration  # Update velocity

        self.si.setStatePropagator(oc.StatePropagatorFn(propagate))

    def set_planner(self, planner_name):
        '''Set the motion planner by name (e.g., RRT, RRTstar, KPIECE1).'''
        planner_map = {
            "RRT": oc.RRT,
            # "RRTConnect": oc.RRTConnect,
            # "RRTstar": oc.RRTstar,
            "KPIECE1": oc.KPIECE1
        }
        planner_cls = planner_map.get(planner_name)
        if planner_cls:
            self.planner = planner_cls(self.ss.getSpaceInformation())
            self.ss.setPlanner(self.planner)
        else:
            print(f"{planner_name} not recognized. Please add it first.")

    def plan_start_goal(self, start, goal, allowed_time=DEFAULT_PLANNING_TIME):
        '''Plan a path from start to goal using control-based planning.'''
        orig_robot_state = self.robot.get_cur_state()

        # Set start and goal states
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # Attempt to solve within the allowed time
        solved = self.ss.solve(allowed_time)
        if solved:
            print(f"Found solution: interpolating path")
            sol_path_control = self.ss.getSolutionPath()
            # For PathControl, use the interpolate method directly without passing a parameter
            sol_path_control.interpolate()  # OMPL will automatically interpolate the path
            sol_path_states = sol_path_control.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            return True, sol_path_list
        else:
            print("No solution found")
            return False, []

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME):
        '''Plan a path to the goal from the current robot state.'''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time)

    def execute(self, path, dynamics=False):
        '''Execute the planned path in PyBullet.'''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i], force=5 * 240)
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)

    def state_to_list(self, state):
        '''Convert OMPL state to a list of joint angles and velocities.'''
        joint_angles = [state[i] for i in range(self.robot.num_dim)]
        joint_velocities = [state[self.robot.num_dim + i] for i in range(self.robot.num_dim)]
        return joint_angles + joint_velocities


class BoxDemo():
    def __init__(self):
        self.obstacles = []

        # Connect to PyBullet with GUI
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / 240.)

        # Set additional search path for PyBullet URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Load the robot model
        robot_id = p.loadURDF("models/franka_description/robots/panda_arm.urdf", (0, 0, 0), useFixedBase=1)
        robot = PbOMPLRobot(robot_id)
        self.robot = robot

        # Set up the pb_ompl interface and planner
        self.pb_ompl_interface = PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("RRTstar")  # Set the planner to RRT*

        # Add obstacles to the environment
        self.add_obstacles()

    def clear_obstacles(self):
        '''Remove all obstacles from the simulation.'''
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        '''Add predefined obstacles to the environment.'''
        self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        '''Add a box obstacle to the environment.'''
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)
        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        '''Run the demo with predefined start and goal configurations.'''
        start = [0, 0, 0, -1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0]  # Example start state with velocities
        goal = [0, 1.5, 0, -0.1, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]  # Example goal state with velocities

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path


if __name__ == '__main__':
    env = BoxDemo()
    env.demo()
