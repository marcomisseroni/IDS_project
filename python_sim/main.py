import numpy as np
import Limo
import sim_data
import conf_limo
import matplotlib.pyplot as plt

N_sim = 100
limo0 = Limo(np.array([0,0,0]), np.array([2,2]))
sim = sim_data("sin", N_sim, conf_limo.dt, 0.01)
for i in range(N_sim):
    # sim of the three target measures
    target0 = sim.absolute_target_pos(i)
    target1 = sim.absolute_target_pos(i)
    target2 = sim.absolute_target_pos(i)
    # computation of the desired limo position
    limo0.desired_pos()

'''
# PLOT
fig, ax = plt.subplots(figsize=(10, 4))
def draw_static():
    ax.plot(x_sol_0[0,:], x_sol_0[1,:], 'x-', label='x0', alpha=0.7)
    ax.plot(x_sol_1[0,:], x_sol_1[1,:], 'x-', label='x1', alpha=0.7)
    ax.plot(x_sol_2[0,:], x_sol_2[1,:], 'x-', label='x2', alpha=0.7)
    ax.plot(x_des_center[0,:], x_des_center[1,:], 'x-', label='x_des_center', alpha=0.7)
    ax.legend()

def update(frame):
    ax.cla()
    draw_static()
    limo_0.plot_robot(x_sol_0[:,frame], limo_0.robot_ray, 'b', fill=0)
    limo_1.plot_robot(x_sol_1[:,frame], limo_1.robot_ray, 'y', fill=0)
    limo_2.plot_robot(x_sol_2[:,frame], limo_2.robot_ray, 'g', fill=0)
    if frame>limo_0.N:
        ax.plot(x_des_center[0,frame-limo_0.N], x_des_center[1,frame-limo_0.N], 'x-')
        circle = Circle(x_des_center[:2,frame-limo_0.N], r, fill=False)
        ax.add_patch(circle)

draw_static()
limo_0.plot_robot(limo_0.x_init, limo_0.robot_ray, 'b')
limo_1.plot_robot(limo_1.x_init, limo_1.robot_ray, 'b')
limo_2.plot_robot(limo_2.x_init, limo_2.robot_ray, 'b')
ani = FuncAnimation(fig, update, frames=N_sim, interval=100)
plt.show()
'''