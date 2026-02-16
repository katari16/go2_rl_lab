import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import deploy_real.deploy_real_isaaclab as deploy_real_isaaclab
import sys

######### To point the path to where unitree_sdk2_python is stored ########
sys.path.append('.')                                                      #
sys.path.append('..')                                                     #
sys.path.append('/home/theo/deploy/unitree_sdk2_python') # TO MODIFY !!!  #
###########################################################################

from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_go.msg import LowState as LowStateRos

msg = None
base_lin_vel_input = [0, 0, 0, 0]

class KalmanOdomListener(Node):
    def __init__(self):
        super().__init__('kalman_odom_listener')
        self.subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',    # topic published by inekf_odom.py
            self.listener_callback,
            10
        )
    
        self.lowstate_publisher = self.create_publisher(
            LowStateRos,
            '/inekf_lowstate', 
            10
        )

        self.timer = self.create_timer(0.005, self.publish_lowstate)
        
    def listener_callback(self, msg):
        # Estimated Vx,Vy,VZ by the Kalman Filter

        global base_lin_vel_input
        base_lin_vel_input[0] = msg.twist.twist.linear.x
        base_lin_vel_input[1] = msg.twist.twist.linear.y
        base_lin_vel_input[2] = msg.twist.twist.linear.z
        base_lin_vel_input[3] = msg.twist.twist.angular.z
    
    # This function transforms the lowstate msg receive by deploy_real_isaaclab.py to the Lowstate standard format
    def publish_lowstate(self):
        if msg is not None:
            ros_msg = LowStateRos()

            # Simple fields
            ros_msg.head = list(msg.head)
            ros_msg.level_flag = msg.level_flag
            ros_msg.frame_reserve = msg.frame_reserve
            ros_msg.sn = list(msg.sn)
            ros_msg.version = list(msg.version)
            ros_msg.bandwidth = msg.bandwidth
            ros_msg.tick = msg.tick
            ros_msg.wireless_remote = list(msg.wireless_remote)
            ros_msg.bit_flag = msg.bit_flag
            ros_msg.adc_reel = msg.adc_reel
            ros_msg.temperature_ntc1 = msg.temperature_ntc1
            ros_msg.temperature_ntc2 = msg.temperature_ntc2
            ros_msg.power_v = msg.power_v
            ros_msg.power_a = msg.power_a
            ros_msg.fan_frequency = list(msg.fan_frequency)
            ros_msg.reserve = msg.reserve
            ros_msg.crc = msg.crc

            # IMU
            ros_msg.imu_state.quaternion = list(msg.imu_state.quaternion)
            ros_msg.imu_state.gyroscope = list(msg.imu_state.gyroscope)
            ros_msg.imu_state.accelerometer = list(msg.imu_state.accelerometer)
            ros_msg.imu_state.rpy = list(msg.imu_state.rpy)
            ros_msg.imu_state.temperature = msg.imu_state.temperature

            # Motor states (20 motors)
            for i in range(20):
                ros_msg.motor_state[i].mode = msg.motor_state[i].mode
                ros_msg.motor_state[i].q = msg.motor_state[i].q
                ros_msg.motor_state[i].dq = msg.motor_state[i].dq
                ros_msg.motor_state[i].ddq = msg.motor_state[i].ddq
                ros_msg.motor_state[i].tau_est = msg.motor_state[i].tau_est
                ros_msg.motor_state[i].q_raw = msg.motor_state[i].q_raw
                ros_msg.motor_state[i].dq_raw = msg.motor_state[i].dq_raw
                ros_msg.motor_state[i].ddq_raw = msg.motor_state[i].ddq_raw
                ros_msg.motor_state[i].temperature = msg.motor_state[i].temperature
                ros_msg.motor_state[i].lost = msg.motor_state[i].lost
                ros_msg.motor_state[i].reserve = list(msg.motor_state[i].reserve)

            # BMS
            ros_msg.bms_state.version_high = msg.bms_state.version_high
            ros_msg.bms_state.version_low = msg.bms_state.version_low
            ros_msg.bms_state.status = msg.bms_state.status
            ros_msg.bms_state.soc = msg.bms_state.soc
            ros_msg.bms_state.current = msg.bms_state.current
            ros_msg.bms_state.cycle = msg.bms_state.cycle
            ros_msg.bms_state.bq_ntc = list(msg.bms_state.bq_ntc)
            ros_msg.bms_state.mcu_ntc = list(msg.bms_state.mcu_ntc)
            ros_msg.bms_state.cell_vol = list(msg.bms_state.cell_vol)

            # Foot force
            ros_msg.foot_force = list(msg.foot_force)
            ros_msg.foot_force_est = list(msg.foot_force_est)

            # Publish
            self.lowstate_publisher.publish(ros_msg)



def main(args=None):
    rclpy.init(args=args)
    node = KalmanOdomListener()
    rclpy.spin(node)   # spin until the node is killed
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 