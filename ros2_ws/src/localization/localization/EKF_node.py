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
        # kalman for the limo and the person
        self.person_ekf = EKF(initial_person_pos, None, None, conf_kalman.Q_p, conf_kalman.dt, AgentType.PERSON)
        self.ekf = EKF(initial_state, conf_kalman.R_rr, conf_kalman.R_rp, conf_kalman.Q, conf_kalman.dt, AgentType.ROBOT)
        # variables to compute dt
        self.last_callback_time = None
        self.actual_callback_time = None

        self.measurement = None
        # id management
        self.ekf.agent_id = args
        EKF.agent_id = 4
        EKF.agent_dims[0] = self.ekf.n
        EKF.agent_dims[1] = self.ekf.n
        EKF.agent_dims[2] = self.ekf.n
        EKF.agent_dims[3] = self.person_ekf.n
        # messages count
        self.state_msg_count = 0
        self.landmark_msg_count = 0
        self.update_msg_count = 0
        # timer to publish the estimated state
        self.state_timer = self.create_timer(conf_kalman.Tp, self.state_timer_callback)
        self.pub_state = self.create_publisher(State, '/state', 10)
        # topic on which the node publishes
        self.pub_info = self.create_publisher(Landmark, '/info', 10)
        self.pub_update = self.create_publisher(Update, '/update', 10)
        # topic subscription
        self.sub_odometry = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.sub_measurement = self.create_subscription(
            Measurement,
            '/measurement',
            self.measurement_callback,
            10)
        self.sub_info = self.create_subscription(
            Landmark,
            '/info',
            self.info_callback,
            10)
        self.sub_update = self.create_subscription(
            Update,
            'update',
            self.update_callback,
            10)
        
    def odometry_callback(self, msg):
        self.get_logger().info('Message received: "%s"' % msg)
        self.last_callback_time = self.actual_callback_time
        self.actual_callback_time = self.get_clock().now()

        if(self.last_callback_time is not None):
            dt = (self.actual_callback_time - self.last_callback_time).nanoseconds * 1e-9
            self.ekf.dt = dt
            self.person_ekf.dt = dt

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.person_ekf.prediction_step(None)
        self.ekf.prediction_step([v, w])
        self.get_logger().info('Receiving: "%s"' % msg)

    def measurement_callback(self, msg):
        if(msg.id_a == self.ekf.agent_id): 
            self.measurement = [msg.x, msg.y, msg.dtheta]

            if(msg.id_b == self.person_ekf.agent_id):
                ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(self.person_ekf.state, self.person_ekf.phi, self.person_ekf.P, self.measurement, self.person_ekf.agent_id, self.person_ekf.agent_type)
                msg_out = Update()
                msg_out.id_a = self.ekf.agent_id
                msg_out.id_b = msg.id_b
                msg_out.dim_a = self.ekf.n
                msg_out.dim_b = self.person_ekf.n
                msg_out.ra = ra
                msg_out.gamma_a = gamma_a
                msg_out.gamma_b = gamma_b
                msg_out.w1 = W1
                msg_out.w2 = W2
                self.pub_update.publish(msg_out)
                self.get_logger().info('Publishing: "%s"' % msg_out)
                self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)
                self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)
                self.update_msg_count += 1

        if(msg.id_b != self.ekf.agent_id): return 
        msg_out = Landmark()
        msg_out.dim = self.ekf.n
        msg_out.state = self.ekf.state
        msg_out.phi = self.ekf.phi
        msg_out.p = self.ekf.P
        self.pub_info.publish(msg_out)
        self.get_logger().info('Publishing: "%s"' % msg_out)
        self.landmark_msg_count += 1

    def info_callback(self, msg):
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
        msg_out.ra = ra
        msg_out.gamma_a = gamma_a
        msg_out.gamma_b = gamma_b
        msg_out.w1 = W1
        msg_out.w2 = W2
        self.pub_update.publish(msg_out)
        self.get_logger().info('Publishing: "%s"' % msg_out)
        self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.id_b)
        self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.id_b)
        self.update_msg_count += 1

    def update_callback(self, msg):
        if(msg.id_a == self.ekf.agent_id): return
        self.ekf.update_step(msg.ra, msg.gamma_a, msg.gamma_b, msg.w1, msg.w2, msg.id_a, msg.id_b)
        self.person_ekf.update_step(msg.ra, msg.gamma_a, msg.gamma_b, msg.w1, msg.w2, msg.id_a, msg.id_b)

    def state_timer_callback(self):
        msg = State()
        msg.x = self.ekf.state[0]
        msg.y = self.ekf.state[1]
        msg.theta = self.ekf.state[2]
        self.pub_state.publish(msg)
        self.get_logger().info('Publishing: "%s"' % self.ekf.state)
        self.state_msg_count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', type=int, help='Integer agent id for this EKF node')
    parser.add_argument('--robot_state', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Initial robot state: x y theta')
    parser.add_argument('--person_state', nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0], help='Initial person state: x y vx vy')
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