from ompl import base as ob
from ompl import geometric as og
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ompl
import numpy as np

# Configuration
SPACE_DIM = 2

obstacles = [
    ((-0.85, -0.15), (-0.45, 0.50)),
    ((-0.30, -0.30), (0.30, 0.30)),
    ((0.30, 0.00), (0.60, 0.60)),
    ((0.00, -0.75), (0.50, -0.50))
]
start_s = (-0.80, 0.80)
goal_s = (0.80, -0.80)

# Check if the state is valid (not inside any obstacle)
def isStateValid(state):
    for o in obstacles:
        if (o[0][0] <= state[0] <= o[1][0]) and (o[0][1] <= state[1] <= o[1][1]):
            return False
    return True

# Retrieve the coordinates of a vertex in the planner data
def getVertexPair(space, vertex):
    reals = ompl.util.vectorDouble()
    if vertex != ob.PlannerData.NO_VERTEX:
        space.copyToReals(reals, vertex.getState())
        return reals[0], reals[1]
    return None, None

# Plot the result
def plot_result(state_space, path_result, planning_data):
    # Extract points from the solution path
    states = path_result.getStates()
    path_point_x = [s[0] for s in states]
    path_point_y = [s[1] for s in states]

    # Collect all edges from the planning data
    edge_list_x, edge_list_y = [], []
    edge_list = ompl.util.vectorUint()
    
    for i in range(planning_data.numVertices()):
        n_edge = planning_data.getEdges(i, edge_list)
        for j in range(n_edge):
            x_1, y_1 = getVertexPair(state_space, planning_data.getVertex(i))
            x_2, y_2 = getVertexPair(state_space, planning_data.getVertex(edge_list[j]))
            edge_list_x.append([x_1, x_2])
            edge_list_y.append([y_1, y_2])

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Plot obstacles
    for o in obstacles:
        rect = patches.Rectangle(o[0], o[1][0] - o[0][0], o[1][1] - o[0][1], edgecolor='#000000', facecolor='#f08080')
        ax.add_patch(rect)

    # Plot edges generated during the planning process
    for i in range(len(edge_list_x)):
        ax.plot(edge_list_x[i], edge_list_y[i], color="#808080", linewidth=0.5)

    # Plot the final path
    ax.plot(path_point_x, path_point_y, color="#ff0000")

    # Plot start and goal points
    ax.scatter(start_s[0], start_s[1], color="#ff8c00", label="Start")
    ax.scatter(goal_s[0], goal_s[1], color="#ff8c00", label="Goal")

    # Show plot
    plt.legend()
    plt.show()
    fig.savefig("result.png")

# Plan the path
def plan():
    # Create a 2D state space
    space = ob.RealVectorStateSpace(SPACE_DIM)

    # Set bounds for the space
    bounds = ob.RealVectorBounds(SPACE_DIM)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # Setup the problem with the space
    ss = og.SimpleSetup(space)

    # Use RRT* planner
    planner = og.RRTstar(ss.getSpaceInformation())
    ss.setPlanner(planner)

    # Set the state validity checker
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # Define start and goal states
    start = ob.State(space)
    start_state = start.get()
    for i in range(SPACE_DIM):
        start_state[i] = start_s[i]

    goal = ob.State(space)
    goal_state = goal.get()
    for i in range(SPACE_DIM):
        goal_state[i] = goal_s[i]

    ss.setStartAndGoalStates(start, goal)

    # Try to solve the problem
    solved = ss.solve(0.1)

    if solved:
        # Simplify the solution path
        ss.simplifySolution()
        path_result = ss.getSolutionPath()

        # Get planner data for plotting
        si = ss.getSpaceInformation()
        pdata = ob.PlannerData(si)
        ss.getPlannerData(pdata)

        # Plot the result
        plot_result(space, path_result, pdata)
    else:
        print("No solution found")

if __name__ == "__main__":
    plan()
