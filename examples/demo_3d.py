import ompl
from ompl import base as ob
from ompl import geometric as og
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

# Constants and configurations
SPACE_DIM = 6
min_bound = -1.5
max_bound = 1.5

robot_x, robot_y, robot_z = 0.3, 0.4, 0.2
robot_r = np.sqrt(robot_x**2 + robot_y**2 + robot_z**2)

obstacles = [(0, 0, 0, 0.75), (0.5, 0.5, 0.5, 0.3), (-0.5, -0.5, 0.5, 0.2)]
start_s = (-0.90, 0.90, 0.90, 0, 0, 0)
goal_s = (0.90, -0.90, -0.90, np.pi / 2, np.pi / 4, 0)

# Check if the state is valid by checking against obstacles
def isStateValid(state):
    for o in obstacles:
        distance = np.sqrt((state[0] - o[0])**2 + (state[1] - o[1])**2 + (state[2] - o[2])**2)
        if distance < robot_r + o[3]:
            return False
    return True

# Retrieve vertex position from the planner data
def getVertexPair(space, vertex):
    reals = ompl.util.vectorDouble()
    if vertex != ob.PlannerData.NO_VERTEX:
        space.copyToReals(reals, vertex.getState())
        return reals[0], reals[1], reals[2]
    return None, None, None

# Draw the bounding box for the robot at a given position and rotation
def draw_box(position, rotation):
    euler = np.array(rotation)
    rot = Rotation.from_euler('XYX', euler)

    edges = [[robot_x, robot_y, robot_z], [robot_x, -robot_y, robot_z], [robot_x, robot_y, -robot_z], [robot_x, -robot_y, -robot_z],
             [-robot_x, robot_y, robot_z], [-robot_x, -robot_y, robot_z], [-robot_x, robot_y, -robot_z], [-robot_x, -robot_y, -robot_z]]
    
    rotated_edges = [rot.apply(np.array(e)) + np.array(position) for e in edges]
    rotated_edges = np.array(rotated_edges)
    
    return [[rotated_edges[0], rotated_edges[1], rotated_edges[3], rotated_edges[2]],
            [rotated_edges[0], rotated_edges[2], rotated_edges[6], rotated_edges[4]],
            [rotated_edges[0], rotated_edges[1], rotated_edges[5], rotated_edges[4]],
            [rotated_edges[1], rotated_edges[3], rotated_edges[7], rotated_edges[5]],
            [rotated_edges[2], rotated_edges[3], rotated_edges[7], rotated_edges[6]],
            [rotated_edges[4], rotated_edges[5], rotated_edges[7], rotated_edges[6]]]

# Draw a sphere for an obstacle at a given position and radius
def draw_sphere(point, r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) + point[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + point[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
    return [x, y, z]

# Plot the result
def plot_result(state_space, path_result, planning_data):
    # Extract waypoints from the solution path
    states = path_result.getStates()
    waypoints = np.array([[s[i] for i in range(SPACE_DIM)] for s in states])

    # Collect edges from the planning data
    edge_list_x, edge_list_y, edge_list_z = [], [], []
    edge_list = ompl.util.vectorUint()
    
    for i in range(planning_data.numVertices()):
        n_edge = planning_data.getEdges(i, edge_list)
        for j in range(n_edge):
            x_1, y_1, z_1 = getVertexPair(state_space, planning_data.getVertex(i))
            x_2, y_2, z_2 = getVertexPair(state_space, planning_data.getVertex(edge_list[j]))
            edge_list_x.append([x_1, x_2])
            edge_list_y.append([y_1, y_2])
            edge_list_z.append([z_1, z_2])

    # Create figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min_bound * 1.5, max_bound * 1.5)
    ax.set_ylim(min_bound * 1.5, max_bound * 1.5)
    ax.set_zlim(min_bound * 1.5, max_bound * 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot obstacles
    for o in obstacles:
        sphere = draw_sphere(o[0:3], o[3])
        ax.plot_surface(sphere[0], sphere[1], sphere[2], color='#ff69b4', alpha=0.5)

    # Plot edges created during exploration
    for i in range(len(edge_list_x)):
        ax.plot(edge_list_x[i], edge_list_y[i], edge_list_z[i], color="#808080", linewidth=0.5)

    # Plot solution path
    tr_waypoints = waypoints.T
    ax.plot(tr_waypoints[0], tr_waypoints[1], tr_waypoints[2], color="#ff0000")
    
    # Plot the robot bounding box along the path
    for point in waypoints:
        pos = point[0:3]
        rot = point[3:6]
        result = draw_box(pos, rot)
        ax.add_collection3d(Poly3DCollection(result, facecolors='#4169e1', linewidths=1, edgecolors='#000000', alpha=0.25))

    # Mark start and goal positions
    ax.scatter3D(start_s[0], start_s[1], start_s[2], color="#ff8c00")
    ax.scatter3D(goal_s[0], goal_s[1], goal_s[2], color="#ff8c00")

    ax.set_box_aspect([1, 1, 1])
    plt.show()
    fig.savefig("result.png")

# Path planning function
def plan():
    # Create a 6D state space (x, y, z, roll, pitch, yaw)
    space = ob.RealVectorStateSpace(SPACE_DIM)

    # Set bounds for position and orientation
    bounds = ob.RealVectorBounds(SPACE_DIM)
    for i in range(SPACE_DIM - 3):
        bounds.setLow(i, min_bound)
        bounds.setHigh(i, max_bound)
    for i in range(3, SPACE_DIM):
        bounds.setLow(i, -np.pi)
        bounds.setHigh(i, np.pi)
    space.setBounds(bounds)

    # Setup planning problem
    ss = og.SimpleSetup(space)

    # Use RRT* planner
    planner = og.RRTstar(ss.getSpaceInformation())
    ss.setPlanner(planner)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # Define start and goal states
    start = ob.State(space)
    goal = ob.State(space)
    for i in range(SPACE_DIM):
        start[i] = start_s[i]
        goal[i] = goal_s[i]

    ss.setStartAndGoalStates(start, goal)

    # Solve the planning problem
    solved = ss.solve(4.0)

    if solved:
        # Simplify and retrieve the solution path
        ss.simplifySolution()
        path_result = ss.getSolutionPath()
        print(path_result)

        # Get planner data and plot the result
        si = ss.getSpaceInformation()
        pdata = ob.PlannerData(si)
        ss.getPlannerData(pdata)

        space = path_result.getSpaceInformation().getStateSpace()
        plot_result(space, path_result, pdata)
    else:
        print("No solution found")

if __name__ == "__main__":
    plan()
