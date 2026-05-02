import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import numpy as np
from limo_control.agent_type import AgentType
from limo_control.localization_system import EKF
import conf_kalman

class ExtendedKalmanFilter(Node):

    def __init__(self, initial_state, initial_person_pos):
        
        super.__init__('extended_kalman_filter')
        self.ekf = EKF(initial_state, conf_kalman.R_rr, conf_kalman.R_rp, conf_kalman.Q, conf_kalman.dt, AgentType.ROBOT)
        self.person_ekf = EKF(initial_person_pos, None, None, conf_kalman.Q_p, conf_kalman.dt, AgentType.PERSON)
        self.last_callback_time = None
        self.actual_callback_time = None
        self.measurement = None

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

    def update_callback(self, msg):
        if(msg.data.id_a == self.ekf.agent_id): return
        self.ekf.update_step(msg.data.ra, msg.data.gamma_a, msg.data.gamma_b, msg.data.W1, msg.data.W2, msg.data.id_a, msg.data.id_b)