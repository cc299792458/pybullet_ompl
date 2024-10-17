import sys
import pybullet as p
import pybullet_data
import os.path as osp

# Add the path to access pb_ompl module
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pb_ompl

class BoxDemo:
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
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        # Set up the pb_ompl interface and planner
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("RRT")  # Set planner to RRT for kinodynamic planning

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
        start = [0, 0, 0, -1, 0, 1.5, 0]
        goal = [0, 1.5, 0, -0.1, 0, 0.2, 0]

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path

if __name__ == '__main__':
    env = BoxDemo()
    env.demo()
