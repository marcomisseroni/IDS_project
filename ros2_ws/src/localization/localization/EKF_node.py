import rclpy
from rclpy.node import Node
import argparse
import sys
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import numpy as np
from localization.agent_type import AgentType
from localization.localization_system import EKF
import conf_kalman

class ExtendedKalmanFilter(Node):

    def __init__(self, initial_state, initial_person_pos, args):
        
        super().__init__('extended_kalman_filter')
        self.person_ekf = EKF(initial_person_pos, None, None, conf_kalman.Q_p, conf_kalman.dt, AgentType.PERSON)
        self.ekf = EKF(initial_state, conf_kalman.R_rr, conf_kalman.R_rp, conf_kalman.Q, conf_kalman.dt, AgentType.ROBOT)
        self.last_callback_time = None
        self.actual_callback_time = None
        self.measurement = None
        self.ekf.agent_id = args
        EKF.agent_id = 4

        self.pub_info = self.create_publisher(Info, '/info', 10)
        self.pub_update = self.create_publisher(Update, '/update', 10)

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
            Info,
            '/info',
            self.info_callback,
            10)
        self.sub_update = self.create_subscription(
            Update,
            'update',
            self.update_callback,
            10)
        
    def odometry_callback(self, msg):
        self.get_logger().info('Message received: "%s"' % msg.data)
        self.last_callback_time = self.actual_callback_time
        self.actual_callback_time = self.get_clock().now()

        if(self.last_callback_time is not None):
            dt = self.actual_callback_time - self.last_callback_time
            self.ekf.dt = dt
            self.person_ekf.dt = dt

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.person_ekf.prediction_step(None)
        self.ekf.prediction_step([v, w])

    def measurement_callback(self, msg):
        if(msg.data.id_a == self.ekf.agent_id): 
            self.measurement = msg.data.measurement

            if(msg.data.id_b == self.person_ekf.agent_id):
                ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(self.person_ekf.state, self.person_ekf.phi, self.person_ekf.P, self.measurement, self.person_ekf.agent_id, self.person_ekf.agent_type)
                msg_out = Update()
                msg_out.data.id_a = self.ekf.agent_id
                msg_out.data.id_b = msg.data.id_b
                msg_out.data.ra = ra
                msg_out.data.gamma_a = gamma_a
                msg_out.data.gamma_b = gamma_b
                msg_out.data.W1 = W1
                msg_out.data.W2 = W2
                self.pub_update.publish(msg_out)
                self.get_logger().info('Publishing: "%s"' % msg_out.data)
                self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)
                self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, self.person_ekf.agent_id)

        if(msg.data.id_b != self.ekf.agent_id): return 
        msg_out = Info()
        msg_out.data.state = self.ekf.state
        msg_out.data.phi = self.ekf.phi
        msg_out.data.P = self.ekf.P
        self.pub_info.publish(msg_out)
        self.get_logger().info('Publishing: "%s"' % msg_out.data)

    def info_callback(self, msg):
        if(msg.data.id_a != self.ekf.agent_id): return
        ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(msg.data.state, msg.data.phi, msg.data.P, self.measurement, msg.data.id_b, msg.data.b_agent_type)
        msg_out = Update()
        msg_out.data.id_a = self.ekf.agent_id
        msg_out.data.id_b = msg.data.id_b
        msg_out.data.ra = ra
        msg_out.data.gamma_a = gamma_a
        msg_out.data.gamma_b = gamma_b
        msg_out.data.W1 = W1
        msg_out.data.W2 = W2
        self.pub_update.publish(msg_out)
        self.get_logger().info('Publishing: "%s"' % msg_out.data)
        self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.data.id_b)
        self.person_ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.data.id_b)

    def update_callback(self, msg):
        if(msg.data.id_a == self.ekf.agent_id): return
        self.ekf.update_step(msg.data.ra, msg.data.gamma_a, msg.data.gamma_b, msg.data.W1, msg.data.W2, msg.data.id_a, msg.data.id_b)
        self.person_ekf.update_step(msg.data.ra, msg.data.gamma_a, msg.data.gamma_b, msg.data.W1, msg.data.W2, msg.data.id_a, msg.data.id_b)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', type=int, help='Integer agent id for this EKF node')
    parser.add_argument('robot_state', nargs=3, type=float, help='Initial robot state: x y theta')
    parser.add_argument('person_state', nargs=4, type=float, help='Initial person state: x y vx vy')
    parsed_args, ros_args = parser.parse_known_args(args[1:])

    rclpy.init(args=ros_args)

    initial_robot_state = np.array(parsed_args.robot_state)
    initial_person_state = np.array(parsed_args.person_state)
    extended_kalman_filter = ExtendedKalmanFilter(initial_robot_state, initial_person_state, parsed_args.agent_id)

    rclpy.spin(extended_kalman_filter)

    extended_kalman_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)