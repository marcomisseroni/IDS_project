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
from pathlib import Path

#  ______ _  ________               _
# |  ____| |/ /  ____|             | |
# | |__  | ' /| |__ _ __   ___   __| | ___
# |  __| |  < |  __| '_ \ / _ \ / _` |/ _ \
# | |____| . \| |  | | | | (_) | (_| |  __/
# |______|_|\_\_|  |_| |_|\___/ \__,_|\___|
#             ______
#            |______|

'''
Node that implements the Extended Kalman Filter for one agent: either a limo
(one instance per robot, selected by its agent_id) or the person (a single
instance, launched with --person). The person is the only single source of
truth for its own state - unlike the robots it is never the "measuring" side
of a measurement (id_a), only the "measured" side (id_b), so the id_a-only
branches below are simply never triggered when running as the person.
'''

ROOT_ROS_WS = Path(__file__).parent.parent.parent.parent.parent.parent.parent
CSV_PATH = ROOT_ROS_WS / "csv_data"

PERSON_ID = 3
ROBOT_DIM = 3
PERSON_DIM = 4

class ExtendedKalmanFilter(Node):

    # fixed set of agent pairs whose cross-covariance block gets its own csv
    # column: limo0, limo1, limo2, person(=PERSON_ID=3), all combinations i<j
    _CROSS_COV_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def __init__(self, agent_id, is_person=False):

        super().__init__('person_ekf_node' if is_person else 'extended_kalman_filter')
        self.is_person = is_person

        if is_person:
            agent_id = PERSON_ID
            initial_state = np.hstack((conf_kalman.target_init, np.array([0.0, 0.0])))
            self.ekf = EKF(initial_state, None, None, conf_kalman.Q_p, conf_kalman.dt, AgentType.PERSON, conf_kalman.P0_p)
        else:
            initial_state = conf_kalman.limo_init[agent_id]
            self.ekf = EKF(initial_state, conf_kalman.R_rr, conf_kalman.R_rp, conf_kalman.Q, conf_kalman.dt, AgentType.ROBOT, conf_kalman.P0_rr)

        # variables to compute dt
        self.last_callback_time = None
        self.actual_callback_time = None

        # csv
        suffix = "person" if is_person else str(agent_id)
        n = self.ekf.n
        state_header = ",".join(f"x{i}" for i in range(n))
        std_header = ",".join(f"std_x{i}" for i in range(n))
        # one column per cross_cov pair (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) - fixed
        # since there are always these 4 agents (limo0, limo1, limo2, person=3)
        pair_header = ",".join(f"cc_{i}_{j}" for i, j in self._CROSS_COV_PAIRS)
        self.csv_file_pred = open(CSV_PATH / f"pred_states_{suffix}.csv", "w")
        self.csv_file_pred.write(
            f"timestamp,dt,{state_header},{std_header},"
            f"trace_P,cross_cov_norm,{pair_header},eig_min_P,sym_err_P,phi_trace,v,w,source\n"
        )
        self.csv_file_upd = open(CSV_PATH / f"upd_states_{suffix}.csv", "w")
        self.csv_file_upd.write(
            f"id_a,id_b,timestamp,{state_header},{std_header},"
            f"trace_P,cross_cov_norm,{pair_header},trace_P_before,eig_min_P,sym_err_P,phi_trace,"
            "correction_norm,residual_norm,residual\n"
        )

        # to start
        # pending measurements this agent still needs to complete, keyed by
        # the id_b of the measured agent (avoids mismatching a measurement
        # with the wrong /info reply if multiple targets are measured
        # in quick succession)
        self.pending_measurements = {}
        # id management
        self.ekf.agent_id = agent_id
        EKF.agent_id = 4
        EKF.agent_dims[0] = ROBOT_DIM
        EKF.agent_dims[1] = ROBOT_DIM
        EKF.agent_dims[2] = ROBOT_DIM
        EKF.agent_dims[PERSON_ID] = PERSON_DIM
        self.ekf._cross_cov_set()
        # messages count
        self.state_msg_count = 0
        self.landmark_msg_count = 0
        self.update_msg_count = 0
        # timer to publish the estimated state
        self.state_timer = self.create_timer(conf_kalman.dt_MPC, self.state_timer_callback)
        # timer that fills in with a zero-velocity prediction whenever the last
        # prediction (real or filler) is older than dt, e.g. odometry missing/too slow
        # (always the case for the person, which has no odometry of its own)
        self.predict_timer = self.create_timer(conf_kalman.dt, self.predict_check_callback)
        self.pub_state = self.create_publisher(State, '/person_state' if is_person else '/limo_state', 10)
        # topic on which the node publishes
        self.pub_info = self.create_publisher(Landmark, '/info', 10)
        self.pub_update = self.create_publisher(Update, '/update', 10)
        # topic subscription
        if not is_person:
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

    def odometry_callback(self, msg):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self._predict(v, w)

    def predict_check_callback(self):
        if self.actual_callback_time is None:
            self._predict(0.0, 0.0, source="filler")
            return
        elapsed = (self.get_clock().now() - self.actual_callback_time).nanoseconds * 1e-9
        if elapsed >= conf_kalman.dt:
            self._predict(0.0, 0.0, source="filler")

    @staticmethod
    def _state_stats(state, P):
        state_str = ",".join(f"{x:.6f}" for x in np.asarray(state).flatten())
        std = np.sqrt(np.abs(np.diag(P)))
        std_str = ",".join(f"{x:.6f}" for x in std)
        trace_P = np.trace(P)
        eig_min = np.linalg.eigvalsh(P).min()
        sym_err = np.linalg.norm(P - P.T)
        return state_str, std_str, trace_P, eig_min, sym_err

    @staticmethod
    def _cross_cov_norm(cross_cov):
        # scalar summary of how correlated this agent's belief is with the
        # rest of the network: sum of the Frobenius norms of every
        # off-diagonal cross-covariance block it is tracking.
        if not cross_cov:
            return 0.0
        return float(sum(np.linalg.norm(block) for block in cross_cov.values()))

    @classmethod
    def _cross_cov_pair_norms_str(cls, cross_cov):
        # one Frobenius norm per pair (not summed), in the fixed order of
        # _CROSS_COV_PAIRS, so every csv row has the same columns
        norms = []
        for pair in cls._CROSS_COV_PAIRS:
            block = cross_cov.get(pair)
            norms.append(np.linalg.norm(block) if block is not None else 0.0)
        return ",".join(f"{x:.6f}" for x in norms)

    def _log_update(self, id_a, id_b, r_a, old_state, old_trace_P, phi_trace_before):
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        state_str, std_str, trace_P, eig_min, sym_err = self._state_stats(self.ekf.state, self.ekf.P)
        cross_cov_norm = self._cross_cov_norm(self.ekf.cross_cov)
        pair_norms_str = self._cross_cov_pair_norms_str(self.ekf.cross_cov)
        correction_norm = np.linalg.norm(self.ekf.state - old_state)
        r_a = np.asarray(r_a).flatten()
        residual_norm = np.linalg.norm(r_a)
        residual_str = ";".join(f"{x:.6f}" for x in r_a)
        self.csv_file_upd.write(
            f"{id_a},{id_b},{timestamp:.6f},{state_str},{std_str},"
            f"{trace_P:.6f},{cross_cov_norm:.6f},{pair_norms_str},{old_trace_P:.6f},{eig_min:.6f},{sym_err:.6f},{phi_trace_before:.6f},"
            f"{correction_norm:.6f},{residual_norm:.6f},{residual_str}\n"
        )

    def _predict(self, v, w, source="odom"):
        self.last_callback_time = self.actual_callback_time
        self.actual_callback_time = self.get_clock().now()

        if(self.last_callback_time is not None):
            dt = (self.actual_callback_time - self.last_callback_time).nanoseconds * 1e-9
            self.ekf.dt = dt

        if self.is_person:
            self.ekf.state[2] = 0.0
            self.ekf.state[3] = 0.0
        self.ekf.prediction_step([v, w])

        timestamp = self.actual_callback_time.nanoseconds * 1e-9
        state_str, std_str, trace_P, eig_min, sym_err = self._state_stats(self.ekf.state, self.ekf.P)
        cross_cov_norm = self._cross_cov_norm(self.ekf.cross_cov)
        pair_norms_str = self._cross_cov_pair_norms_str(self.ekf.cross_cov)
        phi_trace = np.trace(self.ekf.phi)
        self.csv_file_pred.write(
            f"{timestamp:.6f},{self.ekf.dt:.6f},{state_str},{std_str},"
            f"{trace_P:.6f},{cross_cov_norm:.6f},{pair_norms_str},{eig_min:.6f},{sym_err:.6f},{phi_trace:.6f},{v:.6f},{w:.6f},{source}\n"
        )

    def measurement_callback(self, msg):

        if(msg.id_a == self.ekf.agent_id):
            if msg.id_b == PERSON_ID:
                self.pending_measurements[msg.id_b] = np.array([msg.x, msg.y])
            else:
                self.pending_measurements[msg.id_b] = np.array([msg.x, msg.y, msg.dtheta])

        if(msg.id_b == self.ekf.agent_id):
            msg_out = Landmark()
            msg_out.id_a = msg.id_a
            msg_out.id_b = self.ekf.agent_id
            msg_out.dim = self.ekf.n
            msg_out.state = [float(x) for x in self.ekf.state.flatten()]

            msg_out.phi = [float(x) for x in self.ekf.phi.flatten()]
            msg_out.p = [float(x) for x in self.ekf.P.flatten()]
            self.pub_info.publish(msg_out)
            self.landmark_msg_count += 1

    def info_callback(self, msg):
        if(msg.id_a != self.ekf.agent_id): return
        measurement = self.pending_measurements.pop(msg.id_b, None)
        if measurement is None: return
        state_b = msg.state
        phi_b = np.array(msg.phi).reshape((msg.dim, msg.dim))
        P_b = np.array(msg.p).reshape((msg.dim, msg.dim))
        b_agent_type = AgentType.PERSON if msg.id_b == PERSON_ID else AgentType.ROBOT
        ra, gamma_a, gamma_b, W1, W2 = self.ekf.measurement(state_b, phi_b, P_b, measurement, msg.id_b, b_agent_type)
        msg_out = Update()
        msg_out.id_a = self.ekf.agent_id
        msg_out.id_b = msg.id_b
        msg_out.dim_a = self.ekf.n
        msg_out.dim_b = msg.dim
        msg_out.ra = np.asarray(ra, dtype=float).ravel().tolist()
        msg_out.gamma_a = np.asarray(gamma_a, dtype=float).ravel().tolist()
        msg_out.gamma_b = np.asarray(gamma_b, dtype=float).ravel().tolist()
        msg_out.w1 = np.asarray(W1, dtype=float).ravel().tolist()
        msg_out.w2 = np.asarray(W2, dtype=float).ravel().tolist()
        self.pub_update.publish(msg_out)
        #self.get_logger().info('Publishing on update')
        old_state = self.ekf.state.copy()
        old_trace_P = np.trace(self.ekf.P)
        phi_trace_before = np.trace(self.ekf.phi)
        self.ekf.update_step(ra, gamma_a, gamma_b, W1, W2, self.ekf.agent_id, msg.id_b)
        self._log_update(self.ekf.agent_id, msg.id_b, ra, old_state, old_trace_P, phi_trace_before)
        self.update_msg_count += 1


    def update_callback(self, msg):
        if(msg.id_a == self.ekf.agent_id): return

        measurement_dim = len(msg.ra)
        ra = np.asarray(msg.ra, dtype=float)
        gamma_a = np.asarray(msg.gamma_a, dtype=float).reshape((msg.dim_a, measurement_dim))
        gamma_b = np.asarray(msg.gamma_b, dtype=float).reshape((msg.dim_b, measurement_dim))
        w1 = np.asarray(msg.w1, dtype=float).reshape((msg.dim_b, measurement_dim))
        w2 = np.asarray(msg.w2, dtype=float).reshape((msg.dim_a, measurement_dim))

        old_state = self.ekf.state.copy()
        old_trace_P = np.trace(self.ekf.P)
        phi_trace_before = np.trace(self.ekf.phi)
        self.ekf.update_step(ra, gamma_a, gamma_b, w1, w2, msg.id_a, msg.id_b)
        self._log_update(msg.id_a, msg.id_b, ra, old_state, old_trace_P, phi_trace_before)
        #self.get_logger().info('Update callback')

    def state_timer_callback(self):
        msg_out = State()
        msg_out.id = self.ekf.agent_id
        msg_out.x = self.ekf.state[0]
        msg_out.y = self.ekf.state[1]
        msg_out.theta = 0.0 if self.is_person else self.ekf.state[2]
        self.pub_state.publish(msg_out)
        #self.get_logger().info('Publishing: "%s"' % self.ekf.state)
        self.state_msg_count += 1

    def destroy_node(self):
        self.csv_file_pred.close()
        self.csv_file_upd.close()
        super().destroy_node()

def main():
    # strip --ros-args and everything after it before parsing our own
    # arguments, otherwise argparse can mistake a ROS remapping token
    # (e.g. __node:=...) for the optional agent_id positional
    own_args = rclpy.utilities.remove_ros_args(sys.argv)[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', type=int, nargs='?', default=None, help='Integer agent id for this EKF node (omit when using --person)')
    parser.add_argument('--person', action='store_true', help='Run this node as the person agent instead of a robot')
    parser.add_argument('--robot_state', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Initial robot state: x y theta')
    parsed_args = parser.parse_args(own_args)

    if not parsed_args.person and parsed_args.agent_id is None:
        print("Usage: ros2 run localization EKF_node limo_ID [--robot_state x y theta]")
        print("       ros2 run localization EKF_node --person")
        return

    rclpy.init(args=sys.argv)

    extended_kalman_filter = ExtendedKalmanFilter(parsed_args.agent_id, is_person=parsed_args.person)
    try:
        rclpy.spin(extended_kalman_filter)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        extended_kalman_filter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
