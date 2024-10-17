import time
import casadi as ca
import pybullet as p
import pybullet_data

# Initialize PyBullet environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the KUKA robot model
robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])

# Get joint limits and velocity limits from PyBullet for each joint
joint_limits = []
for joint_idx in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, joint_idx)
    joint_limits.append({
        'joint_name': joint_info[1],
        'lower_limit': joint_info[8],        # Lower joint position limit
        'upper_limit': joint_info[9],        # Upper joint position limit
        'max_velocity': joint_info[11]       # Maximum joint velocity
    })

# Define the optimization problem in CasADi
N = 100  # Number of time steps
dt = 1 / 240.  # Time interval for each step

# Define symbolic variables for state (joint positions and velocities) and control (joint accelerations)
q = ca.MX.sym('q', 7)  # 7 joint positions
v = ca.MX.sym('v', 7)  # 7 joint velocities
u = ca.MX.sym('u', 7)  # 7 joint accelerations

# Define CasADi optimization instance
opti = ca.Opti()

# Define optimization variables for the trajectory
Q = opti.variable(7, N+1)  # Joint positions (7 joints, N+1 steps)
V = opti.variable(7, N+1)  # Joint velocities (7 joints, N+1 steps)
U = opti.variable(7, N)    # Joint accelerations (7 joints, N steps)

# Objective function: minimize the squared sum of accelerations (smooth motion)
opti.minimize(ca.sumsqr(U))

# Dynamics constraints: update joint velocity and joint position using acceleration
for k in range(N):
    # Velocity update: v[k+1] = v[k] + u[k] * dt
    opti.subject_to(V[:, k+1] == V[:, k] + dt * U[:, k])
    
    # Position update: q[k+1] = q[k] + v[k] * dt
    opti.subject_to(Q[:, k+1] == Q[:, k] + dt * V[:, k])

# Apply PyBullet's joint limits and velocity constraints to the optimization problem
for joint_idx, limits in enumerate(joint_limits):
    # Position limits
    opti.subject_to(Q[joint_idx, :] >= limits['lower_limit'])  # Joint position lower limit
    opti.subject_to(Q[joint_idx, :] <= limits['upper_limit'])  # Joint position upper limit
    
    # Velocity limits
    opti.subject_to(ca.fabs(V[joint_idx, :]) <= limits['max_velocity'])  # Joint velocity limit
    opti.subject_to(ca.fabs(V[joint_idx, :]) >= -limits['max_velocity'])  # Joint velocity limit

# Initial and final state constraints
opti.subject_to(Q[:, 0] == [0] * 7)  # Initial joint positions (all start at 0)
opti.subject_to(V[:, 0] == [0] * 7)  # Initial joint velocities (all start at 0)
opti.subject_to(Q[:, -1] == [0, 1, 0, 0, 0, 0, 0])  # Desired final joint positions

# Set the solver options and select IPOPT as the solver
opti.solver('ipopt')

# Start timing the optimization
start_time = time.time()

# Solve the optimization problem
sol = opti.solve()

# End timing the optimization
end_time = time.time()
print(f"Optimization solve time: {end_time - start_time} seconds")

# Apply the optimized trajectory to the robot in PyBullet
for k in range(N):
    # Set joint positions based on the solution
    joint_positions = sol.value(Q[:, k])
    for joint_idx, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, joint_idx, pos)  # Apply position to each joint
    p.stepSimulation()  # Step simulation in PyBullet
    time.sleep(dt)  # Wait to simulate real-time motion
