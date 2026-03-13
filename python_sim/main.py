import numpy as np
import Limo
import sim_data
import conf_limo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

N_sim = conf_limo.N_sim
dt = 0.1
r = conf_limo.r_circle

# limo 0
x0_init = [-r,0,0]
limo_0 = Limo.Limo(x0_init, np.array([2,2]))
x_sol_0 = np.zeros((limo_0.mpc.nx,N_sim))
u_sol_0 = np.zeros((limo_0.mpc.nu,N_sim))
limo_0.mpc.create_OCP_problem()

# limo 1
x1_init = [-r*np.cos(60*np.pi/180),r*np.sin(60*np.pi/180),0]
limo_1 = Limo.Limo(x1_init, np.array([2,2]))
x_sol_1 = np.zeros((limo_1.mpc.nx,N_sim))
u_sol_1 = np.zeros((limo_1.mpc.nu,N_sim))
limo_1.mpc.create_OCP_problem()

# limo 2
x2_init = [-r*np.cos(60*np.pi/180),-r*np.sin(60*np.pi/180),0]
limo_2 = Limo.Limo(x2_init, np.array([2,2]))
x_sol_2 = np.zeros((limo_2.mpc.nx,N_sim))
u_sol_2 = np.zeros((limo_2.mpc.nu,N_sim))
limo_2.mpc.create_OCP_problem()

sim = sim_data.data_sim("sin", N_sim, dt, 0.01)

# Warm start
sol0 = limo_0.mpc.warm_start(x0_init, x1_init, x2_init, r)
sol1 = limo_1.mpc.warm_start(x1_init, x0_init, x2_init, r)
sol2 = limo_2.mpc.warm_start(x2_init, x0_init, x1_init, r)

p0 = np.zeros((2, N_sim))
p1 = np.zeros((2, N_sim))
p2 = np.zeros((2, N_sim))
t = np.zeros((2, N_sim))
c = np.zeros((2, N_sim))

for i in range(N_sim):
    # sim of the three target measures
    target0 = sim.absolute_target_pos(i)
    target1 = sim.absolute_target_pos(i)
    target2 = sim.absolute_target_pos(i)
    t[:,i] = (target0 + target1 + target2) / 3
    # computation of the desired limo position
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x0_des = limo_0.desired_pos(target0, target1, target2, np.array([0, 0]), np.array([0, 0]))
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x1_des = limo_1.desired_pos(target0, target1, target2, np.array([0, 0]), np.array([0, 0]))
    p0[:,i], p1[:,i], p2[:,i], c[:,i], x2_des = limo_2.desired_pos(target0, target1, target2, np.array([0, 0]), np.array([0, 0]))
    # MPC for each limo to compute inputs for desired position
    in0 = limo_0.mpc_sim(limo_1.ekf.state[:2], limo_2.ekf.state[:2], x0_des)
    in1 = limo_1.mpc_sim(limo_0.ekf.state[:2], limo_2.ekf.state[:2], x1_des)
    in2 = limo_2.mpc_sim(limo_0.ekf.state[:2], limo_1.ekf.state[:2], x2_des)
    # ekf
    enc0 = sim.sensors_from_input(in0)
    enc1 = sim.sensors_from_input(in1)
    enc2 = sim.sensors_from_input(in2)
    lid_meas0 = sim.ext_sensors(limo_0.ekf.state)
    lid_meas1 = sim.ext_sensors(limo_1.ekf.state)
    lid_meas2 = sim.ext_sensors(limo_2.ekf.state)
    limo_0.ekf_step(enc0, in0[1], lid_meas0)
    limo_1.ekf_step(enc1, in1[1], lid_meas1)
    limo_2.ekf_step(enc2, in2[1], lid_meas2)


# PLOT
fig, ax = plt.subplots(figsize=(10, 4))
def draw_static():
    ax.plot(t[0,:], t[1,:], 'x-', alpha=0.7)
    ax.legend()

def update(frame):
    ax.cla()
    draw_static()
    ax.plot(t[0,frame],  t[1,frame],  'o-', label='target', alpha=1)
    ax.plot(c[0,frame],  c[1,frame],  'x-', alpha=0.5)
    ax.plot(p0[0,frame], p0[1,frame], 'o-', label='limo 0', alpha=0.5)
    ax.plot(p1[0,frame], p1[1,frame], 'o-', label='limo 1', alpha=0.5)
    ax.plot(p2[0,frame], p2[1,frame], 'o-', label='limo 2', alpha=0.5)
    circle = Circle(c[:2,frame], conf_limo.r_circle, fill=False)
    ax.add_patch(circle)
    ax.set_xlim([1, 10])
    ax.set_ylim([-1, 3])
    plt.grid(True)
    ax.legend()
    plt.gca().set_aspect('equal')

draw_static()
ani = FuncAnimation(fig, update, frames=N_sim, interval=100)
plt.show()