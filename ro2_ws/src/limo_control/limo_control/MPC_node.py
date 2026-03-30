import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
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
                                                                                                       
    def __init__(self, x_init, dt_mpc):
        super().__init__('mpc')
        # states and inputs size
        self.nx = 3 # size of state vector x = [x,y,theta]
        self.nu = 2 # size of control vector u = [v, w]

        # MPC settings
        self.dt_MPC = self.declare_parameter('dt', 0.0).value # time step MPC
        self.N = self.declare_parameter('N', 0.0).value  # time horizon MPC
        self.N_sim = self.declare_parameter('N_sim', 0.0).value # number of simulation steps
        v_min = self.declare_parameter('v_min', 0.0).value
        v_max = self.declare_parameter('v_max', 0.0).value
        w_min = self.declare_parameter('w_min', 0.0).value
        w_max = self.declare_parameter('w_max', 0.0).value

        # create the dynamics function
        x = cs.SX.sym('x', self.nx)
        u = cs.SX.sym('u', self.nu)
        # defining the dynamics
        rhs = cs.vertcat(cs.cos(x[-1])*u[0],
                         cs.sin(x[-1])*u[0],
                         u[1])
        self.f = cs.Function('f', [x, u], [rhs])

        self.x_init = x_init               # initial state
        self.u_min = [v_min, w_min]   # minimum value of u
        self.u_max = [v_max, w_max]   # maximum value of u
        self.robot_ray = self.declare_parameter('r_collision', 0.0).value      # ray of the robot used for visualization
        self.sol = np.zeros(self.N_sim)

        # OCP weigths
        self.w_p = 1e4 # final position weigth
        self.w_v = 1e-2 # velocity weight
        self.w_final_v = 1e-2 # final velocity cost weight
        self.w_a = 1e-1 # weigth on the angle of the limo in respect to the target

        # limo positions
        self.x1 = np.zeros(3)
        self.x2 = np.zeros(3)
        self.x3 = np.zeros(3)

        # Target position
        self.target = np.zeros(2)

        # Flag 
        self.start = False

        # Communication stuff
        # Pub
        self.publisher = self.create_publisher(Float64MultiArray, 'control_inputs', 10)
        self.timer = self.create_timer(self.dt_MPC, self.timer_callback)

        # Sub
        self.subscription_1 = self.create_subscription(Float64MultiArray, '/limo1/ekf_state', self.ekf_listener_callback1, 10)
        self.subscription_2 = self.create_subscription(Float64MultiArray, '/limo2/ekf_state', self.ekf_listener_callback2, 10)
        self.subscription_3 = self.create_subscription(Float64MultiArray, '/limo3/ekf_state', self.ekf_listener_callback3, 10)
        self.subscription_a = self.create_subscription(String, 'admin', self.admin_listener_callback, 10)
        self.subscription_t = self.create_subscription(Float64MultiArray, 'target', self.target_listener_callback, 10)
        #self.subscription_1
        #self.subscription_2
        #self.subscription_3
        #self.subscription_a

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
        if not self.start:
            return
        self.sol, state = self.MPC_step(self.sol, self.x_init, self.x_des, self.x1, self.x2, self.robot_ray, self.target)
        control_inputs = Float64MultiArray()
        control_inputs.data = self.sol.value(self.U[0])
        self.publisher.publish(control_inputs)
        self.get_logger().info('Publishing: "%s"' % control_inputs.data)

    def ekf_listener_callback1(self, msg):
        # Save the EKF result and use it for the MPC step
        self.x1 = np.array(msg.data)
        return

    def ekf_listener_callback2(self, msg):
        # Save the EKF result and use it for the MPC step
        self.x2 = np.array(msg.data)
        return

    def ekf_listener_callback3(self, msg):
        # Save the EKF result and use it for the MPC step
        self.x3 = np.array(msg.data)
        return

    def admin_listener_callback(self, msg):
        # Listen to the admin topic to start or stop the MPC
        if msg.data == "start":
            self.start = True
        elif msg.data == "stop":
            self.start = False
        return
    
    def target_listener_callback(self, msg):
        # Listen to the target topic to update the target position
        self.target = np.array(msg.data)
        return

def main(args=None):
    rclpy.init(args=args)

    # Initialize the MPC
    x0 = np.zeros(3)
    mpc = MPC(x_init=x0, dt=0.3)
    x_sol_1 = np.zeros((mpc.nx,mpc.N_sim))
    u_sol_1 = np.zeros((mpc.nu,mpc.N_sim))
    mpc.create_OCP_problem()
    sol0 = mpc.warm_start(x0, mpc.x2, mpc.x3, mpc.robot_ray)
    
    rclpy.spin(mpc)
    mpc.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()