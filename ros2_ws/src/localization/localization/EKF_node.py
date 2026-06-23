import numpy as np
# ros2 stuff
import rclpy
from rclpy.node import Node
# to have arguments in the node call
import argparse
import sys
# to use EKF class
from localization.agent_type import AgentType
from localization.localization_system import EKF
# configuration file containing convariances, ...
from limo_description import conf_limo as conf_kalman
# message types used
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from project_interfaces.msg import Measurement
from project_interfaces.msg import Landmark
from project_interfaces.msg import Update
from project_interfaces.msg import State

#  ______ _  ________               _      
# |  ____| |/ /  ____|             | |     
# | |__  | ' /| |__ _ __   ___   __| | ___ 
# |  __| |  < |  __| '_ \ / _ \ / _` |/ _ \
# | |____| . \| |  | | | | (_) | (_| |  __/
# |______|_|\_\_|  |_| |_|\___/ \__,_|\___|
#             ______                       
#            |______|                      

'''
Node that implements the Extended Kalman Filter.
It includes both the localization of the limo and the person, so each Limo has also the 
information about the person.
'''

class ExtendedKalmanFilter(Node):

    def __init__(self, initial_state, initial_person_pos, args):
        
        super().__init__('extended_kalman_filter')
        initial_state = conf_kalman.limo_init[args]
        initial_person_pos = np.hstack((conf_kalman.target_init,np.array([0.0,0.0])))
        # kalman for the limo and the person
        self.person_ekf = EKF(initial_person_pos, None, None, conf_kalman.Q_p, conf_kalman.dt, AgentType.PERSON)
        self.ekf = EKF(initial_state, conf_kalman.R_rr, conf_kalman.R_rp, conf_kalman.Q, conf_kalman.dt, AgentType.ROBOT)
        # variables to compute dt
        self.last_callback_time = None
        self.actual_callback_time = None
        
        # to start
        self.is_running = False
        self.measurement = None
        # id management
        self.ekf.agent_id = args
        self.person_ekf.agent_id = 3
        EKF.agent_id = 4
        EKF.agent_dims[0] = self.ekf.n
        EKF.agent_dims[1] = self.ekf.n
        EKF.agent_dims[2] = self.ekf.n
        EKF.agent_dims[3] = self.person_ekf.n
        self.ekf._cross_cov_set()
        self.person_ekf._cross_cov_set()
        # messages count
        self.state_msg_count = 0
        self.landmark_msg_count = 0
        self.update_msg_count = 0
        # timer to publish the estimated state
        self.state_timer = self.create_timer(conf_kalman.dt_MPC, self.state_timer_callback)
        self.pub_limo_state = self.create_publisher(State, '/limo_state', 10)
        self.pub_person_state = self.create_publisher(State, '/person_state', 10)
        # topic on which the node publishes
        self.pub_info = self.create_publisher(Landmark, '/info', 10)
        self.pub_update = self.create_publisher(Update, '/update', 10)
        # topic subscription
        self.sub_odometry = self.create_subscription(
            Odometry,
            'odom',
            self.odometry_callback,
            10)
        self.sub_measurement = self.create_subscription(
            Measurement,
            '/measurement_routed',
            self.measurement_callback,
            10)
        self.sub_info = self.create_subscription(
            Landmark,
            '/info',
            self.info_callback,
            10)
        self.sub_update = self.create_subscription(
            Update,
            '/update',
            self.update_callback,
            10)
        self.sub_admin = self.create_subscription(
            String,
            '/admin',
            self.admin_callback,
            10)
        
    def odometry_callback(self, msg):
        if(not self.is_running): return
        self.last_callback_time = self.actual_callback_time
        self.actual_callback_time = self.get_clock().now()

        if(self.last_callback_time is not None):
            dt = (self.actual_callback_time - self.last_callback_time).nanoseconds * 1e-9
            self.ekf.dt = dt
            self.person_ekf.dt = dt

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.person_ekf.state[2] = 0.0
        self.person_ekf.state[3] = 0.0
        self.person_ekf.prediction_step(None)
        self.ekf.prediction_step([v, w])
        #self.get_logger().info(f'Receiving odometry message: v={v}, w={w}')

    def measurement_callback(self, msg):
        if(not self.is_running): return
        if(msg.x == conf_kalman.x_camera): return
        if(msg.id_a == self.ekf.agent_id): 
            if msg.id_b != self.person_ekf.agent_id:
                self.measurement = np.array([msg.x, msg.y, msg.dtheta])
            else:
                self.measurement = np.array([msg.x, msg.y])
                ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(self.person_ekf.state, self.person_ekf.phi, self.person_ekf.P, self.measurement, self.person_ekf.agent_id, self.person_ekf.agent_type)
                msg_out = Update()
                msg_out.id_a = self.ekf.agent_id
                msg_out.id_b = msg.id_b
                msg_out.dim_a = self.ekf.n
                msg_out.dim_b = self.person_ekf.n
                msg_out.ra = np.asarray(ra, dtype=float).ravel().tolist()
                msg_out.gamma_a = np.asarray(gamma_a, dtype=float).ravel().tolist()
                msg_out.gamma_b = np.asarray(gamma_b, dtype=float).ravel().tolist()
                msg_out.w1 = np.asarray(W1, dtype=float).ravel().tolist()
                msg_out.w2 = np.asarray(W2, dtype=float).ravel().tolist()
                self.pub_update.publish(msg_out)
                self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)
                self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)
                self.update_msg_count += 1
                #self.get_logger().info(f'Measuring person position: x={msg.x}, y={msg.y}')

        if(msg.id_b != self.ekf.agent_id): return 
        msg_out = Landmark()
        msg_out.dim = self.ekf.n
        msg_out.state = self.ekf.state
        print("-------------CHECK FOR ERROR----------------")
        print(type(self.ekf.state))
        print(self.ekf.state.shape)
        print(self.ekf.state)
        msg_out.phi = self.ekf.phi
        msg_out.p = self.ekf.P
        self.pub_info.publish(msg_out)
        #self.get_logger().info('Publishing state on pub_info: "%s"' % msg_out.state)
        self.landmark_msg_count += 1

    def info_callback(self, msg):
        if(not self.is_running): return
        if(msg.id_a != self.ekf.agent_id): return
        state_b = msg.state
        phi_b = msg.phi.reshape((msg.dim, msg.dim))
        P_b = msg.p.reshape((msg.dim, msg.dim))
        ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(state_b, phi_b, P_b, self.measurement, msg.id_b, AgentType.ROBOT)
        msg_out = Update()
        msg_out.id_a = self.ekf.agent_id
        msg_out.id_b = msg.id_b
        msg_out.dim_a = self.ekf.n
        msg_out.dim_b = self.ekf.n
        msg_out.ra = np.asarray(ra, dtype=float).ravel().tolist()
        msg_out.gamma_a = np.asarray(gamma_a, dtype=float).ravel().tolist()
        msg_out.gamma_b = np.asarray(gamma_b, dtype=float).ravel().tolist()
        msg_out.w1 = np.asarray(W1, dtype=float).ravel().tolist()
        msg_out.w2 = np.asarray(W2, dtype=float).ravel().tolist()
        self.pub_update.publish(msg_out)
        #self.get_logger().info('Publishing on update')
        self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.id_b)
        self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.id_b)
        self.update_msg_count += 1

    def update_callback(self, msg):
        if(not self.is_running): return
        if(msg.id_a == self.ekf.agent_id): return

        measurement_dim = len(msg.ra)
        ra = np.asarray(msg.ra, dtype=float)
        gamma_a = np.asarray(msg.gamma_a, dtype=float).reshape((msg.dim_a, measurement_dim))
        gamma_b = np.asarray(msg.gamma_b, dtype=float).reshape((msg.dim_b, measurement_dim))
        w1 = np.asarray(msg.w1, dtype=float).reshape((msg.dim_b, measurement_dim))
        w2 = np.asarray(msg.w2, dtype=float).reshape((msg.dim_a, measurement_dim))

        self.ekf.update_step(ra, gamma_a, gamma_b, w1, w2, msg.id_a, msg.id_b)
        self.person_ekf.update_step(ra, gamma_a, gamma_b, w1, w2, msg.id_a, msg.id_b)
        #self.get_logger().info('Update callback')

    def state_timer_callback(self):
        if(not self.is_running): return
        msg_limo = State()
        msg_limo.id = self.ekf.agent_id
        msg_limo.x = self.ekf.state[0]
        msg_limo.y = self.ekf.state[1]
        msg_limo.theta = self.ekf.state[2]
        msg_person = State()
        msg_person.id = 3
        msg_person.x = self.person_ekf.state[0]
        msg_person.y = self.person_ekf.state[1]
        msg_person.theta = 0.0
        self.pub_limo_state.publish(msg_limo)
        self.pub_person_state.publish(msg_person)
        #self.get_logger().info('Publishing: "%s"' % self.ekf.state)
        #self.get_logger().info('Publishing: "%s"' % self.person_ekf.state)
        self.state_msg_count += 1

    def admin_callback(self, msg):
        if(msg.data == 'start_ekf'):
            self.is_running = True
        elif(msg.data == 'stop_ekf'):
            self.is_running = False
        else:
            print('ERROR: invalid command')
            print('Usage: "start_ekf" to start the EKF_node, "stop_ekf" to stop the EKF_node')

def main():
    if len(sys.argv) <= 1:
        print("Usage: ros2 run localization EKF_node limo_ID [--robot_state x y theta] [--person_state x y vx vy]")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', type=int, help='Integer agent id for this EKF node')
    parser.add_argument('--robot_state', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Initial robot state: x y theta')
    parser.add_argument('--person_state', nargs=4, type=float, default=[0.0, -3.0, 0.0, 0.0], help='Initial person state: x y vx vy')
    parsed_args, ros_args = parser.parse_known_args(sys.argv[1:])

    rclpy.init(args=ros_args)

    initial_robot_state = np.array(parsed_args.robot_state)
    initial_person_state = np.array(parsed_args.person_state)
    extended_kalman_filter = ExtendedKalmanFilter(initial_robot_state, initial_person_state, parsed_args.agent_id)
    rclpy.spin(extended_kalman_filter)
    extended_kalman_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()