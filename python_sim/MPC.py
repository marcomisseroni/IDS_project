import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
import time
from matplotlib.patches import Rectangle
import conf_limo as limo

class MPC:
#  _____       _ _   _       _ _          _   _             
# |_   _|     (_) | (_)     | (_)        | | (_)            
#   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __  
#   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#  _| |_| | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|                                                          
                                                                                                       
    def __init__(self, x_init):
        # states and inputs size
        self.nx = 3 # size of state vector x = [x,y,theta]
        self.nu = 2 # size of control vector u = [v, w]

        # MPC settings
        self.dt_MPC = 0.010 # time step MPC
        self.N = 10  # time horizon MPC
        self.N_sim = 1000 # number of total simulation steps

        # create the dynamics function
        x = cs.SX.sym('x', self.nx)
        u = cs.SX.sym('u', self.nu)
        # defining the dynamics
        rhs = cs.vertcat(cs.cos(x[-1])*u[0],
                         cs.sin(x[-1])*u[0],
                         u[1])
        self.f = cs.Function('f', [self.x, self.u], [rhs])

        self.x_init = x_init               # initial state
        self.u_min = [limo.v_min, limo.w_min]   # minimum value of u
        self.u_max = [limo.v_max, limo.w_max]   # maximum value of u
        self.robot_ray = limo.r_collision       # ray of the robot used for visualization

        # OCP weigths
        self.w_p = 1 # final position weigth
        self.w_v = 1e-2 # velocity weight
        self.w_final_v = 1e-2 # final velocity cost weight

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
            self.opti.subject_to(self.X[k+1] == self.X[k] + self.dt * self.f(self.X[k], self.U[k]))

        # Terminal cost
        cost += self.w_p * (self.X[-1] - self.param_x_des).T @ (self.X[-1] - self.param_x_des)
        # Final velocity cost
        cost += self.w_final_v * self.X[-1].T @ self.X[-1]

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
    def warm_start(self, x_des):
        # Solve the problem to convergence the first time
        self.opti.set_value(self.param_x_des, x_des)
        self.opti.set_value(self.param_x_init, self.x_init)
        sol = self.opti.solve()
        return sol

#  _____  _       _       
# |  __ \| |     | |      
# | |__) | | ___ | |_ ___ 
# |  ___/| |/ _ \| __/ __|
# | |    | | (_) | |_\__ \
# |_|    |_|\___/ \__|___/
                         
    def plot_robot(x, ray, color1='r', color2='k', fill=1, axis=None):
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

                      
                                                           