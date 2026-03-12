import numpy as np
import Limo
import sim_data
import conf_limo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

N_sim = conf_limo.N_sim
dt = 0.1
limo0 = Limo.Limo(np.array([0,0,0]), np.array([2,2]))
sim = sim_data.data_sim("sin", N_sim, dt, 0.01)

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
    p0[:,i], p1[:,i], p2[:,i], c[:,i], s_des = limo0.desired_pos(target0, target1, target2, np.array([0, 0]), np.array([0, 0]))


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