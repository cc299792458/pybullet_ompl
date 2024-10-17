from ompl import base as ob
from ompl import control as oc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuration
SPACE_DIM = 2
MAX_POSITION = 1
MAX_VELOCITY = 0.5
MAX_ACCELERATION = 1.0

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
    # Extract position (x, y) from the state using direct indexing
    x = state[0]  # x position
    y = state[1]  # y position

    # Check if the position is inside any obstacle
    for o in obstacles:
        if (o[0][0] <= x <= o[1][0]) and (o[0][1] <= y <= o[1][1]):
            return False
    return True

# Define a propagator that simulates the system's motion under control
def propagate(state, control, duration, result):
    # Simple Euler integration for the motion model
    pos = np.array([state[0], state[1]])  # Extract position
    vel = np.array([state[2], state[3]])  # Extract velocity

    # Update position and velocity using control inputs (acceleration)
    accel = np.array([control[0], control[1]])
    vel = np.clip(vel + accel * duration, -MAX_VELOCITY, MAX_VELOCITY)
    pos += np.clip(vel * duration, -MAX_POSITION, MAX_POSITION)

    # Set new position and velocity in the result
    result[0] = pos[0]  # x position
    result[1] = pos[1]  # y position
    result[2] = vel[0]  # vx velocity
    result[3] = vel[1]  # vy velocity

# Plotting function (adapted for controls with position and velocity)
def plot_result(state_space, path_result, planning_data):
    # Extract positions (x, y) from the solution path
    states = path_result.getStates()
    path_point_x = [s[0] for s in states]  # x position
    path_point_y = [s[1] for s in states]  # y position

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Plot obstacles
    for o in obstacles:
        rect = patches.Rectangle(o[0], o[1][0] - o[0][0], o[1][1] - o[0][1], edgecolor='#000000', facecolor='#f08080')
        ax.add_patch(rect)

    # Plot the final path
    ax.plot(path_point_x, path_point_y, color="#ff0000")

    # Plot start and goal points
    ax.scatter(start_s[0], start_s[1], color="#ff8c00", label="Start")
    ax.scatter(goal_s[0], goal_s[1], color="#ff8c00", label="Goal")

    # Show plot
    plt.legend()
    plt.show()

# Plan the path with dynamics
def plan_with_dynamics():
    # Create a 4D state space: (x, y, vx, vy)
    space = ob.RealVectorStateSpace(SPACE_DIM * 2)

    # Set bounds for the full space (position and velocity)
    bounds = ob.RealVectorBounds(SPACE_DIM * 2)
    
    # Set bounds for position (first 2 dimensions)
    bounds.setLow(0, -MAX_POSITION)
    bounds.setHigh(0, MAX_POSITION)
    bounds.setLow(1, -MAX_POSITION)
    bounds.setHigh(1, MAX_POSITION)
    
    # Set bounds for velocity (next 2 dimensions)
    bounds.setLow(2, -MAX_VELOCITY)
    bounds.setHigh(2, MAX_VELOCITY)
    bounds.setLow(3, -MAX_VELOCITY)
    bounds.setHigh(3, MAX_VELOCITY)

    # Apply the bounds to the space
    space.setBounds(bounds)
    
    # Create the control space (acceleration in x and y directions)
    cspace = oc.RealVectorControlSpace(space, SPACE_DIM)
    
    # Set bounds for controls (acceleration)
    bounds_ctrl = ob.RealVectorBounds(SPACE_DIM)
    bounds_ctrl.setLow(-MAX_ACCELERATION)
    bounds_ctrl.setHigh(MAX_ACCELERATION)
    cspace.setBounds(bounds_ctrl)

    # Create a simple setup object
    ss = oc.SimpleSetup(cspace)
    
    # Set the state validity checker
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    
    # Set the state propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))

    # Define start and goal states (including velocity)
    start = ob.State(space)
    start_state = start.get()
    start_state[0] = start_s[0]
    start_state[1] = start_s[1]
    start_state[2] = 0.0  # Initial velocity in x
    start_state[3] = 0.0  # Initial velocity in y

    goal = ob.State(space)
    goal_state = goal.get()
    goal_state[0] = goal_s[0]
    goal_state[1] = goal_s[1]
    goal_state[2] = 0.0  # Final velocity in x
    goal_state[3] = 0.0  # Final velocity in y

    ss.setStartAndGoalStates(start, goal, 0.01)

    # Use SSTT planner for dynamic systems
    planner = oc.SST(ss.getSpaceInformation())
    ss.setPlanner(planner)

    # Try to solve the problem
    solved = ss.solve(100.0)

    if solved:
        path_result = ss.getSolutionPath()

        # Plot the result
        plot_result(space, path_result, None)
    else:
        print("No solution found")

if __name__ == "__main__":
    plan_with_dynamics()
