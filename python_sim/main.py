import numpy as np
import random
import Limo
import sim_data
import conf_limo
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from localization.agent_type import AgentType
from localization.localization_system import EKF

print("Starting simulation...")

N_sim = conf_limo.N_sim
dt = 0.1
r = conf_limo.r_circle
sim = sim_data.data_sim("sin", N_sim, dt, 0.01)
target_init = np.array([sim.global_target_pos(0)])
center_global = np.array([0.0 ,0.0])
flag_rel_meas = 0
mu = 0
sigma = 0.01

print("Initializing limo0")

x0_init = [r,0,0]
limo_0 = Limo.Limo(x0_init, target_init)
x_sol_0 = np.zeros((limo_0.mpc.nx,N_sim))
u_sol_0 = np.zeros((limo_0.mpc.nu,N_sim))
limo_0.mpc.create_OCP_problem()
limo_0_real = EKF(x0_init, conf_limo.R_rr, conf_limo.R_rp, conf_limo.Q, dt, AgentType.ROBOT)

print("Initializing limo 1...")

x1_init = [-r*np.cos(60*np.pi/180),r*np.sin(60*np.pi/180),0]
limo_1 = Limo.Limo(x1_init, target_init)
x_sol_1 = np.zeros((limo_1.mpc.nx,N_sim))
u_sol_1 = np.zeros((limo_1.mpc.nu,N_sim))
limo_1.mpc.create_OCP_problem()
limo_1_real = EKF(x1_init, conf_limo.R_rr, conf_limo.R_rp, conf_limo.Q, dt, AgentType.ROBOT)

print("Initializing limo 2...")

x2_init = [-r*np.cos(60*np.pi/180),-r*np.sin(60*np.pi/180),0]
limo_2 = Limo.Limo(x2_init, target_init)
x_sol_2 = np.zeros((limo_2.mpc.nx,N_sim))
u_sol_2 = np.zeros((limo_2.mpc.nu,N_sim))
limo_2.mpc.create_OCP_problem()
limo_2_real = EKF(x2_init, conf_limo.R_rr, conf_limo.R_rp, conf_limo.Q, dt, AgentType.ROBOT)

print("Initializing person aget")

Q_p = np.zeros((4, 4))
Q_p[0, 0] = dt**4 / 4
Q_p[0, 2] = dt**3 / 2
Q_p[1, 1] = dt**4 / 4
Q_p[1, 3] = dt**3 / 2
Q_p[2, 0] = dt**3 / 2
Q_p[2, 2] = dt**2
Q_p[3, 1] = dt**3 / 2
Q_p[3, 3] = dt**2
person_initial_state = np.append(target_init, np.array([0, 0]))
person = EKF(person_initial_state, None, None, Q_p, dt, AgentType.PERSON) # R matrix not used for person

print("Warm start for the MPC")

# Warm start
limo_0.sol = limo_0.mpc.warm_start(x0_init, x1_init, x2_init, conf_limo.r_collision, target_init)
limo_1.sol = limo_1.mpc.warm_start(x1_init, x0_init, x2_init, conf_limo.r_collision, target_init)
limo_2.sol = limo_2.mpc.warm_start(x2_init, x0_init, x1_init, conf_limo.r_collision, target_init)

p0 = np.zeros((2, N_sim))
p1 = np.zeros((2, N_sim))
p2 = np.zeros((2, N_sim))
t = np.zeros((2, N_sim))
c = np.zeros((2, N_sim))
state0 = np.zeros((3, N_sim))
state1 = np.zeros((3, N_sim))
state2 = np.zeros((3, N_sim))
state0_real = np.zeros((3, N_sim))
state1_real = np.zeros((3, N_sim))
state2_real = np.zeros((3, N_sim))
input0 = np.zeros((2, N_sim))
input1 = np.zeros((2, N_sim))
input2 = np.zeros((2, N_sim))
person_state = np.zeros((2, N_sim))
cov0 = np.zeros(N_sim)
cov1 = np.zeros(N_sim)
cov2 = np.zeros(N_sim)
cov0_xy = np.zeros((N_sim, 2, 2))
cov1_xy = np.zeros((N_sim, 2, 2))
cov2_xy = np.zeros((N_sim, 2, 2))
person_cov_xy = np.zeros((N_sim, 2, 2))
ccov0 = np.zeros(N_sim)
ccov1 = np.zeros(N_sim)
ccov2 = np.zeros(N_sim)

def plot_cov_ellipse(cov, mean, n_std=2.0, ax=None, color='tab:blue', alpha=0.2, label=None):

    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(xy=mean,
                  width=width,
                  height=height,
                  angle=angle,
                  edgecolor=color,
                  facecolor='none',
                  linewidth=2,
                  alpha=alpha,
                  label=label)

    ax.add_patch(ell)
    ax.scatter(*mean, c=color, s=20)

    return ell
    

print("Starting MPC loop...")

for i in range(N_sim):
    print("iteration ", i, " / ", N_sim)
    print("Simulate camera readings...")
    # sim of the target measures
    target = person.state[:2]
    t[:,i] = sim.global_target_pos(i)
    print("Compute desired positions for each limo...")
    # computation of the desired limo position
    #center_global += limo_0.frame_movement
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x0_des = limo_0.desired_pos(target-center_global, target-center_global, target-center_global, limo_1.ekf.state, limo_2.ekf.state)
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x1_des = limo_1.desired_pos(target-center_global, target-center_global, target-center_global, limo_0.ekf.state, limo_2.ekf.state)
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x2_des = limo_2.desired_pos(target-center_global, target-center_global, target-center_global, limo_0.ekf.state, limo_1.ekf.state)
    print("Perform MPC step...")
    # MPC for each limo to compute inputs for desired position
    in0 = limo_0.mpc_sim(limo_1.ekf.state, limo_2.ekf.state, x0_des)
    in1 = limo_1.mpc_sim(limo_0.ekf.state, limo_2.ekf.state, x1_des)
    in2 = limo_2.mpc_sim(limo_0.ekf.state, limo_1.ekf.state, x2_des)
    limo_0_real.prediction_step(in0)
    limo_1_real.prediction_step(in1)
    limo_2_real.prediction_step(in2)
    # ekf
    print("Prediction step of the EKF...")
    epsilon1 = random.gauss(mu, sigma)
    epsilon2 = random.gauss(mu, sigma)
    prop_unc = np.array([epsilon1, epsilon2])
    limo_0.ekf.prediction_step(in0 + prop_unc)
    limo_1.ekf.prediction_step(in1 + prop_unc)
    limo_2.ekf.prediction_step(in2 + prop_unc)
    person.prediction_step(None)
    print("Simulate relative measurement and perform EKF step...\n")
    epsilon1 = random.gauss(mu, sigma)
    epsilon2 = random.gauss(mu, sigma)
    epsilon3 = random.gauss(mu, sigma)
    meas_unc = np.array([epsilon1, epsilon2, epsilon3])
  
    if(flag_rel_meas == 0): # limo_0 measures person
        meas = sim.sim_robot_person_meas(limo_0_real.state, sim.global_target_pos(i), np.array([epsilon1, epsilon2]))
        r_a, gamma_a, gamma_b, W1, W2 = limo_0.ekf.measurement(person.state, person.phi, person.P, meas, person.agent_id, AgentType.PERSON)
        id_a = limo_0.ekf.agent_id
        id_b = person.agent_id
    elif(flag_rel_meas == 1): # limo_1 measures limo_0
        meas = sim.sim_robot_robot_meas(limo_1_real.state, limo_0_real.state, meas_unc)
        r_a, gamma_a, gamma_b, W1, W2 = limo_1.ekf.measurement(limo_0.ekf.state, limo_0.ekf.phi, limo_0.ekf.P, meas, limo_0.ekf.agent_id, AgentType.ROBOT)
        id_a = limo_1.ekf.agent_id
        id_b = limo_0.ekf.agent_id
    elif(flag_rel_meas == 2): # limo_2 measures person
        meas = sim.sim_robot_person_meas(limo_2_real.state, sim.global_target_pos(i), np.array([epsilon1, epsilon2]))
        r_a, gamma_a, gamma_b, W1, W2 = limo_2.ekf.measurement(person.state, person.phi, person.P, meas, person.agent_id, AgentType.PERSON)
        id_a = limo_2.ekf.agent_id
        id_b = person.agent_id
    elif(flag_rel_meas == 3): # limo_2 measures limo_0
        meas = sim.sim_robot_robot_meas(limo_2_real.state, limo_0_real.state, meas_unc)
        r_a, gamma_a, gamma_b, W1, W2 = limo_2.ekf.measurement(limo_0.ekf.state, limo_0.ekf.phi, limo_0.ekf.P, meas, limo_0.ekf.agent_id, AgentType.ROBOT)
        id_a = limo_2.ekf.agent_id
        id_b = limo_0.ekf.agent_id
    elif(flag_rel_meas == 4): # limo_1 measures person
        meas = sim.sim_robot_person_meas(limo_1_real.state, sim.global_target_pos(i), np.array([epsilon1, epsilon2]))
        r_a, gamma_a, gamma_b, W1, W2 = limo_1.ekf.measurement(person.state, person.phi, person.P, meas, person.agent_id, AgentType.PERSON)
        id_a = limo_1.ekf.agent_id
        id_b = person.agent_id

    flag_rel_meas += 1
    if flag_rel_meas > 4: flag_rel_meas = 0
    limo_0.ekf.update_step(r_a, gamma_a, gamma_b, W1, W2, id_a, id_b)
    limo_1.ekf.update_step(r_a, gamma_a, gamma_b, W1, W2, id_a, id_b)
    limo_2.ekf.update_step(r_a, gamma_a, gamma_b, W1, W2, id_a, id_b)
    person.update_step(r_a, gamma_a, gamma_b, W1, W2, id_a, id_b)

    # frame update
    #limo_0.frame_update()
    #limo_1.frame_update()
    #limo_2.frame_update()
    #limo_0_real.state[:2] -= limo_0.frame_movement
    #limo_1_real.state[:2] -= limo_1.frame_movement
    #limo_2_real.state[:2] -= limo_2.frame_movement
    # frame update for person
    #person.state[:2] -= limo_0.frame_movement
    # data to plot
    p0[:,i] += center_global
    p1[:,i] += center_global
    p2[:,i] += center_global
    c[:,i] += center_global
    x0 = limo_0.ekf.state[0] + center_global[0]
    y0 = limo_0.ekf.state[1] + center_global[1]
    theta0 = limo_0.ekf.state[2]
    x1 = limo_1.ekf.state[0] + center_global[0]
    y1 = limo_1.ekf.state[1] + center_global[1]
    theta1 = limo_1.ekf.state[2]
    x2 = limo_2.ekf.state[0] + center_global[0]
    y2 = limo_2.ekf.state[1] + center_global[1]
    theta2 = limo_2.ekf.state[2]

    state0[:,i] = np.array([x0, y0, theta0])
    state1[:,i] = np.array([x1, y1, theta1])
    state2[:,i] = np.array([x2, y2, theta2])
    person_state[:, i] = person.state[:2]
    # inputs
    input0[:,i] = in0
    input1[:,i] = in1
    input2[:,i] = in2
    # self covariance
    cov0[i] = np.trace(limo_0.ekf.P)
    cov1[i] = np.trace(limo_1.ekf.P)
    cov2[i] = np.trace(limo_2.ekf.P)
    cov0_xy[i] = limo_0.ekf.P[:2, :2]
    cov1_xy[i] = limo_1.ekf.P[:2, :2]
    cov2_xy[i] = limo_2.ekf.P[:2, :2]
    person_cov_xy[i] = person.P[:2, :2]
    # cross covariance
    ccov0[i] = np.linalg.norm(limo_0.ekf.cross_cov[0, 1] + limo_0.ekf.cross_cov[0, 2] + limo_0.ekf.cross_cov[1, 2])
    ccov1[i] = np.linalg.norm(limo_1.ekf.cross_cov[0, 1] + limo_1.ekf.cross_cov[0, 2] + limo_1.ekf.cross_cov[1, 2])
    ccov2[i] = np.linalg.norm(limo_2.ekf.cross_cov[0, 1] + limo_2.ekf.cross_cov[0, 2] + limo_2.ekf.cross_cov[1, 2])

# PLOT
fig, ax = plt.subplots(figsize=(10, 4))
txt0 = ax.text(0.02, 0.95, '', transform=ax.transAxes)
txt1 = ax.text(0.02, 0.90, '', transform=ax.transAxes)
txt2 = ax.text(0.02, 0.85, '', transform=ax.transAxes)
def draw_static():
    ax.plot(t[0,:], t[1,:], 'x-', alpha=0.7)

def update(frame):
    ax.cla()
    draw_static()
    ax.plot(t[0,frame],  t[1,frame],  'o-', label='target', alpha=1)
    ax.plot(c[0,frame],  c[1,frame],  'x-', alpha=0.5)
    ax.plot(p0[0,frame], p0[1,frame], 'o-', alpha=0.5)
    ax.plot(p1[0,frame], p1[1,frame], 'o-', alpha=0.5)
    ax.plot(p2[0,frame], p2[1,frame], 'o-', alpha=0.5)
    circle = Circle(c[:2,frame], conf_limo.r_circle, fill=False)
    ax.add_patch(circle)
    # robot positions
    limo_0.mpc.plot_robot(state0[:,frame], limo_0.mpc.robot_ray, 'b', fill=0)
    limo_1.mpc.plot_robot(state1[:,frame], limo_1.mpc.robot_ray, 'y', fill=0)
    limo_2.mpc.plot_robot(state2[:,frame], limo_2.mpc.robot_ray, 'g', fill=0)
    plot_cov_ellipse(cov0_xy[frame], state0[:2, frame], ax=ax, color='b')
    plot_cov_ellipse(cov1_xy[frame], state1[:2, frame], ax=ax, color='y')
    plot_cov_ellipse(cov2_xy[frame], state2[:2, frame], ax=ax, color='g')
    plot_cov_ellipse(person_cov_xy[frame], person_state[:, frame], ax=ax, color='m')
    # text with input results
    txt0 = ax.text(0.02, 0.95, f'u0 = {input0[0,frame]:.2f}', transform=ax.transAxes)
    txt1 = ax.text(0.02, 0.90, f'u1 = {input1[0,frame]:.2f}', transform=ax.transAxes)
    txt2 = ax.text(0.02, 0.85, f'u2 = {input2[0,frame]:.2f}', transform=ax.transAxes)
    plt.grid(True)
    ax.legend()
    plt.gca().set_aspect('equal')

draw_static()
ani = FuncAnimation(fig, update, frames=N_sim, interval=100)
plt.show()

# Additional plots
time = np.arange(N_sim) * dt
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(time, state0[0], label='limo0')
plt.plot(time, state1[0], label='limo1')
plt.plot(time, state2[0], label='limo2')
plt.ylabel('x')
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.plot(time, state0[1])
plt.plot(time, state1[1])
plt.plot(time, state2[1])
plt.ylabel('y')
plt.grid()

plt.subplot(3,1,3)
plt.plot(time, state0[2])
plt.plot(time, state1[2])
plt.plot(time, state2[2])
plt.ylabel('theta')
plt.xlabel('time')
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(time, cov0, label='limo0')
plt.plot(time, cov1, label='limo1')
plt.plot(time, cov2, label='limo2')
plt.ylabel('Trace(P)')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(time, ccov0, label = 'limo0')
plt.plot(time, ccov1, label = 'limo1')
plt.plot(time, ccov2, label = 'limo2')
plt.ylabel('Cross-covariance')
plt.xlabel('time')
plt.grid()

plt.tight_layout()
plt.show()