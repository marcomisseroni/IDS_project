import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
import time
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import conf_limo as limo

class MPC:
#  _____       _ _   _       _ _          _   _             
# |_   _|     (_) | (_)     | (_)        | | (_)            
#   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __  
#   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#  _| |_| | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|                                                          
                                                                                                       
    def __init__(self, x_init, dt):
        # states and inputs size
        self.nx = 3 # size of state vector x = [x,y,theta]
        self.nu = 2 # size of control vector u = [v, w]

        # MPC settings
        self.dt_MPC = dt # time step MPC
        self.N = 10  # time horizon MPC

        # create the dynamics function
        x = cs.SX.sym('x', self.nx)
        u = cs.SX.sym('u', self.nu)
        # defining the dynamics
        rhs = cs.vertcat(cs.cos(x[-1])*u[0],
                         cs.sin(x[-1])*u[0],
                         u[1])
        self.f = cs.Function('f', [x, u], [rhs])

        self.x_init = x_init               # initial state
        self.u_min = [limo.v_min, limo.w_min]   # minimum value of u
        self.u_max = [limo.v_max, limo.w_max]   # maximum value of u
        self.robot_ray = limo.r_collision       # ray of the robot used for visualization

        # OCP weigths
        self.w_p = 1e4 # final position weigth
        self.w_v = 1e-2 # velocity weight
        self.w_final_v = 1e-2 # final velocity cost weight
        self.w_a = 1e1 # weigth on the angle of the limo in respect to the target

#   ____   _____ _____             _               
#  / __ \ / ____|  __ \           | |              
# | |  | | |    | |__) |  ___  ___| |_ _   _ _ __  
# | |  | | |    |  ___/  / __|/ _ \ __| | | | '_ \ 
# | |__| | |____| |      \__ \  __/ |_| |_| | |_) |
#  \____/ \_____|_|      |___/\___|\__|\__,_| .__/ 
#                                           | |    
#                                           |_|    

    def create_OCP_problem(self):
        self.opti = cs.Opti()
        self.param_x_init = self.opti.parameter(self.nx)
        self.param_x_des = self.opti.parameter(self.nx)
        self.param_r = self.opti.parameter(1)
        # parameters for the other two limo positions
        self.param_x1 = self.opti.parameter(self.nx)
        self.param_x2 = self.opti.parameter(self.nx)

        # Create all decision variables for state and control
        self.X, self.U = [], []
        for k in range(self.N+1): 
            self.X += [self.opti.variable(self.nx)]
        for k in range(self.N): 
            self.U += [self.opti.variable(self.nu)]
            self.opti.subject_to(self.opti.bounded(self.u_min, self.U[-1], self.u_max))

        # Add initial conditions
        self.opti.subject_to(self.X[0] == self.param_x_init)

        # Add cost function and dynamics constraints
        cost = 0
        for k in range(self.N):
            # control penality  
            cost += self.w_v * self.U[k].T @ self.U[k]
            # dynamics contraint
            self.opti.subject_to(self.X[k+1] == self.X[k] + self.dt_MPC * self.f(self.X[k], self.U[k]))

        # cost on the desired position (only x and y components)
        cost += self.w_p * (self.X[k][:2] - self.param_x_des[:2]).T @ (self.X[k][:2] - self.param_x_des[:2])
        
        # Final velocity cost
        cost += self.w_final_v * self.X[-1].T @ self.X[-1]

        # no collision constraint
        self.opti.subject_to((self.X[1][0]-self.param_x1[0])**2 + (self.X[1][1]-self.param_x1[1])**2 > (2*self.param_r)**2)
        self.opti.subject_to((self.X[1][0]-self.param_x2[0])**2 + (self.X[1][1]-self.param_x2[1])**2 > (2*self.param_r)**2)

        self.opti.minimize(cost)

        # create the optimization solver
        self.opts = {
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-4,
            "ipopt.compl_inf_tol": 1e-4,
            "print_time": 0,                # print information about execution time
            "detect_simple_bounds": True,
            "ipopt.max_iter": 1000 # only for the warm start
        }
        self.opti.solver("ipopt", self.opts)

    # warm starting the solver to 
    def warm_start(self, x_des, x1, x2, r):
        # Solve the problem to convergence the first time
        self.opti.set_value(self.param_x_des, x_des)
        self.opti.set_value(self.param_x_init, self.x_init)
        self.opti.set_value(self.param_r, r)
        # other two limo initial position
        self.opti.set_value(self.param_x1, x1)
        self.opti.set_value(self.param_x2, x2)
        sol = self.opti.solve()
        # setting the max iter to the lower number for all other optimizations
        self.opts["ipopt.max_iter"] = 10
        self.opti.solver("ipopt", self.opts)
        return sol

                                                                                                   
    def MPC_step(self, sol, x, x_des, x1, x2, r):
        # use current solution as initial guess for next problem
        for t in range(self.N):
            # the values are shifted by 1 timestep
            self.opti.set_initial(self.X[t], sol.value(self.X[t+1]))
        for t in range(self.N-1):
            self.opti.set_initial(self.U[t], sol.value(self.U[t+1]))
        # last value since it is missing is taken as the last value of the previous simulation
        self.opti.set_initial(self.X[self.N], sol.value(self.X[self.N]))
        self.opti.set_initial(self.U[self.N-1], sol.value(self.U[self.N-1]))

        # use the same lagrange multiplier in the next simulation
        lam_g0 = sol.value(self.opti.lam_g)
        self.opti.set_initial(self.opti.lam_g, lam_g0)
        
        self.opti.set_value(self.param_x_init, x)
        self.opti.set_value(self.param_x_des, x_des)
        self.opti.set_value(self.param_r, r)
        self.opti.set_value(self.param_x1, x1)
        self.opti.set_value(self.param_x2, x2)
        try:
            new_sol = self.opti.solve()
        except:
            new_sol = self.opti.debug

        # updating the states value (feedback from kalman filter)
        x = new_sol.value(self.X[1])
        return new_sol, x
    
#  _____  _       _       
# |  __ \| |     | |      
# | |__) | | ___ | |_
# |  ___/| |/ _ \| __/
# | |    | | (_) | |_
# |_|    |_|\___/ \__|
                         
    def plot_robot(self, x, ray, color1='r', color2='k', fill=1, axis=None):
        # x: 3d vector containing x, y and theta of the robot
        # ray: length of the ray of the robot
        # color1: color for the circle
        # color2: color for the rectangle
        # fill: if 1 the circle is filled with the color
        px, py, theta = x[0], x[1], x[2]
        if(axis is None):
            axis = plt.gca()
        axis.add_patch(plt.Circle((px, py), ray, color=color1, fill=fill))
        axis.add_patch(Rectangle((px, py-0.25*ray), ray, 0.5*ray, 
                                angle=theta*180/np.pi, rotation_point=(px, py), 
                                fill=1, color=color2))
        plt.grid(True)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        axis.axis('equal')

#  _______        _   
# |__   __|      | |  
#    | | ___  ___| |_ 
#    | |/ _ \/ __| __|
#    | |  __/\__ \ |_ 
#    |_|\___||___/\__|

if __name__ == "__main__":

    N_sim = limo.N_sim
    # raidus of the circle around the central position
    r = limo.r_circle
    dt = 0.01

    # limo 0
    x0_init = [-r,0,0]
    limo_0 = MPC(x0_init,dt)
    x_sol_0 = np.zeros((limo_0.nx,N_sim))
    u_sol_0 = np.zeros((limo_0.nu,N_sim))
    limo_0.create_OCP_problem()

    # limo 1
    x1_init = [-r*np.cos(60*np.pi/180),r*np.sin(60*np.pi/180),0]
    limo_1 = MPC(x1_init,dt)
    x_sol_1 = np.zeros((limo_1.nx,N_sim))
    u_sol_1 = np.zeros((limo_1.nu,N_sim))
    limo_1.create_OCP_problem()

    # limo 2
    x2_init = [-r*np.cos(60*np.pi/180),-r*np.sin(60*np.pi/180),0]
    limo_2 = MPC(x2_init,dt)
    x_sol_2 = np.zeros((limo_2.nx,N_sim))
    u_sol_2 = np.zeros((limo_2.nu,N_sim))
    limo_2.create_OCP_problem()

    # first desired positions and warm start
    x_des_0_init = [x0_init[0]+0.1, x0_init[1], 0]
    x_des_1_init = [x1_init[0]+0.1, x1_init[1], 0]
    x_des_2_init = [x2_init[0]+0.1, x2_init[1], 0]
    sol0 = limo_0.warm_start(x_des_0_init, x1_init, x2_init, r)
    sol1 = limo_1.warm_start(x_des_1_init, x0_init, x2_init, r)
    sol2 = limo_2.warm_start(x_des_2_init, x0_init, x1_init, r)

    # initializing the states
    x0 = x0_init
    x1 = x1_init
    x2 = x2_init

    x_des_center = np.zeros((3,N_sim))

    # MPC loop
    for i in range(N_sim):
        # desired central position
        x_des_center[:,i] = [0.1+0.005*i, 0.2*np.cos(i/50), 0]

        #solving the MPC step
        sol0, x0 = limo_0.MPC_step(sol0, x0, x_des_center[:,i], x1, x2, r)
        sol1, x1 = limo_1.MPC_step(sol1, x1, x_des_center[:,i], x0, x2, r)
        sol2, x2 = limo_2.MPC_step(sol2, x2, x_des_center[:,i], x0, x1, r)

        # saving the results for plotting
        x_sol_0[:,i] = sol0.value(limo_0.X[0])
        u_sol_0[:,i] = sol0.value(limo_0.U[0])
        x_sol_1[:,i] = sol1.value(limo_1.X[0])
        u_sol_1[:,i] = sol1.value(limo_1.U[0])
        x_sol_2[:,i] = sol2.value(limo_2.X[0])
        u_sol_2[:,i] = sol2.value(limo_2.U[0])
        print("step",i,"inputs:",u_sol_0[:,i],u_sol_1[:,i],u_sol_2[:,i])


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