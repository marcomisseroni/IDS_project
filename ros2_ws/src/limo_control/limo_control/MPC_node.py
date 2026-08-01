import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from project_interfaces.msg import State
from project_interfaces.msg import MPCprediction
from project_interfaces.msg import Desired
from limo_control.MPC_class import MPC
from message_filters import Subscriber
from limo_description import conf_limo
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import String
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import sys
import time

# admin commands must never be missed: reliable delivery + transient_local so a
# command published before this node starts (or while it's restarting) is still
# delivered as soon as it comes up.
ADMIN_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

class MPC_node(Node):

    def __init__(self, id):
        super().__init__('MPC_node')
        # subscibed topics
        self.limo_sub = self.create_subscription(State, '/limo_state', self.limo_states_callback, 10)
        self.target_sub = self.create_subscription(State, '/person_state', self.target_states_callback, 10)
        self.admin_sub = self.create_subscription(String, '/admin', self.admin_callback, ADMIN_QOS)
        # timer for MPC callback every dt
        self.mpc_timer = self.create_timer(conf_limo.dt_MPC, self.MPC_callback)
        # publishing topic
        self.pub_input = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_predictions = self.create_publisher(MPCprediction, 'mpc_prediction', 10)
        self.pub_pos_des = self.create_publisher(Desired, 'desired', 10)

        self.start = False

        # assigning the id's
        ids = [0, 1, 2]
        ids.remove(id)
        self.id = id
        self.id_1 = ids[0]
        self.id_2 = ids[1]

        # buffer vector with latest:
        # - self state information
        self.state = conf_limo.limo_init[self.id]
        # - target state information
        self.target = conf_limo.target_init
        self.prev_target = self.target
        # - other limo number 1 state information
        self.limo_1 = conf_limo.limo_init[self.id_1]
        self.limo_2 = conf_limo.limo_init[self.id_2]
        # - center of the formation
        self.center = conf_limo.initial_center

        # MPC object
        self.mpc_obj = MPC(self.state, conf_limo.dt_MPC)
        self.mpc_obj.create_OCP_problem()

        self.warmstart = True # to keep track if we need to do the warm start or not
        self.sol = None


    def target_states_callback(self, msg):
        # updating the target estimated position with the new informations (filtered data for smoother control)
        last = np.array([msg.x, msg.y])
        prev = self.prev_target
        # update rate
        alpha = 0.15

        self.prev_target = self.target
        self.target = alpha*last + (1-alpha)*prev
    
    def limo_states_callback(self, msg):
        # depending on the limo that sends the message i need to update the buffers
        val = np.array([msg.x, msg.y, msg.theta])
        if(msg.id == self.id):
            self.state = val
        elif(msg.id == self.id_1):
            self.limo_1 = val
        elif(msg.id == self.id_2):
            self.limo_2 == val

    def MPC_callback(self):
        # checking if the MPC has to start
        if(not self.start): return

        # computing the desired position for current step
        des_pos = self.desired_pos()
        if(self.warmstart):
            self.warmstart = False
            self.sol = self.mpc_obj.warm_start(des_pos, self.limo_1, self.limo_2, conf_limo.r_collision, self.target)
        else:
            self.sol = self.mpc_obj.MPC_step(
                self.sol, self.state, des_pos, self.limo_1, self.limo_2, conf_limo.r_collision, self.target)

        # publishing the inputs
        inputs = self.sol.value(self.mpc_obj.U[0])
        msg = Twist()
        msg.linear.x = inputs[0]
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = inputs[1]
        self.pub_input.publish(msg)

        # publishing the predicted states
        pred_states = np.zeros((conf_limo.N, 3))
        for i in range(conf_limo.N):
            pred_states[i,:] = self.sol.value(self.mpc_obj.X[i])
        msg = MPCprediction()
        msg.x = pred_states[:,0].tolist()
        msg.y = pred_states[:,1].tolist()
        msg.theta = pred_states[:,2].tolist()
        #self.pub_predictions.publish(msg)
        
    def admin_callback(self, msg):
        # used to start/stop the mpc
        if(msg.data == 'start_mpc'):
            self.start = True
        elif(msg.data == 'stop_mpc'):
            self.start = False


#  _____            _              _                   
# |  __ \          (_)            | |                  
# | |  | | ___  ___ _ _ __ ___  __| |  _ __   ___  ___ 
# | |  | |/ _ \/ __| | '__/ _ \/ _` | | '_ \ / _ \/ __|
# | |__| |  __/\__ \ | | |  __/ (_| | | |_) | (_) \__ \
# |_____/ \___||___/_|_|  \___|\__,_| | .__/ \___/|___/
#                                     | |              
#                                     |_|              

    def desired_pos(self):
        old_center = np.array([(self.state[0] + self.limo_1[0] +  self.limo_2[0]) / 3, (self.state[1] + self.limo_1[1] +  self.limo_2[1]) / 3])

        # ------------- new center position --------------------
        # angle between previous center and new target
        alpha = np.arctan2(self.target[1]-old_center[1], self.target[0]-old_center[0])
        # distance to move the center
        d = np.sqrt( (self.target[1]-old_center[1])**2 + (self.target[0]-old_center[0])**2 ) - conf_limo.dist

        self.center = np.array([old_center[0]+d*np.cos(alpha), old_center[1]+d*np.sin(alpha)])

        # -------------- position of each limo -----------------
        # three possible positions
        # - p0: along the target direction
        alpha0 = np.arctan2(self.target[1]-self.center[1], self.target[0]-self.center[0])
        p0 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha0), self.center[1]+conf_limo.r_circle*np.sin(alpha0)])
        # - p1: rotated by 120° clockwise
        alpha1 = alpha0 + np.pi*2/3
        p1 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha1), self.center[1]+conf_limo.r_circle*np.sin(alpha1)])
        # - p2: rotated by 120° counterclockwise
        alpha2 = alpha0 - np.pi*2/3
        p2 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha2), self.center[1]+conf_limo.r_circle*np.sin(alpha2)])

        msg = Desired()
        msg.x0 = p0[0]
        msg.y0 = p0[1]
        msg.x1 = p1[0]
        msg.y1 = p1[1]
        msg.x2 = p2[0]
        msg.y2 = p2[1]
        self.pub_pos_des.publish(msg)

        # -------------- choice of the position of each limo -----------------
        positions = np.array([p0, p1, p2])
        limo_positions = np.array([self.state[:2], self.limo_1[:2], self.limo_2[:2]])
        cost_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                cost_matrix[i, j] = np.linalg.norm(limo_positions[i] - positions[j])

        # solve the assignment problem: rows vector contains the limo indices, cols vector contains the position indices
        # example: rows = [0, 1, 2], cols = [2, 0, 1] so limo0 --> p2, limo1 --> p0, limo2 --> p1
        rows, cols = linear_sum_assignment(cost_matrix)
        
        # returning the desired position (x,y,0)
        return np.array([positions[cols[0]][0], positions[cols[0]][1], 0])
        #return np.array([self.target[0], self.target[1], 0])

 
def main(args=None):
    rclpy.init()
    node = MPC_node(int(sys.argv[1]))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
