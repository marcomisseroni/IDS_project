import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from adam.casadi.computations import KinDynComputations
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

#  __  __ _____   _____                   _      
# |  \/  |  __ \ / ____|                 | |     
# | \  / | |__) | |       _ __   ___   __| | ___ 
# | |\/| |  ___/| |      | '_ \ / _ \ / _` |/ _ \
# | |  | | |    | |____  | | | | (_) | (_| |  __/
# |_|  |_|_|     \_____| |_| |_|\___/ \__,_|\___|
                                                
                                                

class MPC(Node):
#  _____       _ _   _       _ _          _   _             
# |_   _|     (_) | (_)     | (_)        | | (_)            
#   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __  
#   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#  _| |_| | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|                                                          
                                                                                                       
    def __init__(self, x_init, dt):
        super().__init__('mpc')
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
        self.w_a = 1e-1 # weigth on the angle of the limo in respect to the target

        # Communication stuff
        # Pub
        self.publisher = self.create_publisher(Float64MultiArray, 'mpc_result', 10)
        self.timer = self.create_timer(self.dt_MPC, self.timer_callback)

        # Sub
        self.subscription_1 = self.create_subscription(Float64MultiArray, 'ekf_result1', self.ekf_listener_callback1, 10)
        self.subscription_2 = self.create_subscription(Float64MultiArray, 'ekf_result2', self.ekf_listener_callback2, 10)
        self.subscription_3 = self.create_subscription(Float64MultiArray, 'ekf_result3', self.ekf_listener_callback3, 10)
        self.subscription_a = self.create_subscription(String, 'admin', self.admin_listener_callback, 10)
        self.subscription_1
        self.subscription_2
        self.subscription_3
        self.subscription_a

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
        self.param_x_target = self.opti.parameter(2)
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
        # cost on the desired angle
        cost += self.w_a * (np.arctan2(self.param_x_target[1] - self.X[1][1], self.param_x_target[0] - self.X[1][0]) - self.X[1][2])**2
        
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
    def warm_start(self, x_des, x1, x2, r, target):
        # Solve the problem to convergence the first time
        self.opti.set_value(self.param_x_des, x_des)
        self.opti.set_value(self.param_x_init, self.x_init)
        self.opti.set_value(self.param_x_target, target)
        self.opti.set_value(self.param_r, r)
        # other two limo initial position
        self.opti.set_value(self.param_x1, x1)
        self.opti.set_value(self.param_x2, x2)
        sol = self.opti.solve()
        # setting the max iter to the lower number for all other optimizations
        self.opts["ipopt.max_iter"] = 10
        self.opti.solver("ipopt", self.opts)
        return sol

                                                                                                   
    def MPC_step(self, sol, x, x_des, x1, x2, r, target):
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
        self.opti.set_value(self.param_x_target, target)
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
    

    def timer_callback(self):
        #MPC step and publish the result
        return

    def ekf_listener_callback1(self, msg):
        # Save the EKF result and use it for the MPC step
        return

    def ekf_listener_callback2(self, msg):
        # Save the EKF result and use it for the MPC step
        return

    def ekf_listener_callback3(self, msg):
        # Save the EKF result and use it for the MPC step
        return

    def admin_listener_callback(self, msg):
        # Listen to the admin topic to start or stop the MPC
        return

def main(args=None):
    rclpy.init(args=args)
    mpc = MPC(x_init=np.zeros(3), dt=0.1)

    # Initialize the MPC

    rclpy.spin(mpc)
    mpc.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()