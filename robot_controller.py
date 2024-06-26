import rospy
from geometry_msgs.msg import Vector3, Transform
import numpy as np
from scipy.spatial.transform import Rotation as R

import time


class RobotController():
    def __init__(self):
        self.robot_ee_frame_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.update_pos_or)
        self.pub_tip_vel = rospy.Publisher('/eyerobot2/desiredTipVelocities', Vector3, queue_size = 3)
        self.pub_tip_vel_angular = rospy.Publisher('/eyerobot2/desiredTipVelocitiesAngular', Vector3, queue_size = 3)
        rospy.sleep(0.5)
        self.position = []
        self.orientation = []
        
    def update_pos_or(self, data):
        x = data.translation.x
        y = data.translation.y
        z = data.translation.z
        rx = data.rotation.x
        ry = data.rotation.y
        rz = data.rotation.z
        rw = data.rotation.w
        self.position = np.array([x, y, z])
        self.orientation = np.array([rx, ry, rz, rw])

    def move_forward_needle_axis(self, kp_linear_vel=2, linear_vel=0.1, duration_sec=1):
        current_quat = self.orientation
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, -1)))
        send_linear_velocity = moving_direction * kp_linear_vel * linear_vel

        for _ in range(int(duration_sec / 0.1)):
            self.pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2])
            rospy.sleep(0.1)
        self.pub_tip_vel.publish(0, 0, 0)
        rospy.sleep(0.1)

    def move_backward_needle_axis(self, kp_linear_vel=2, linear_vel=0.1, duration_sec=1):
        current_quat = self.orientation
        r_current = R.from_quat(current_quat)
        rotation_matrix_current = r_current.as_matrix()
        moving_direction = np.matmul(rotation_matrix_current, np.array((0, 0, 1)))
        send_linear_velocity = moving_direction * kp_linear_vel * linear_vel

        start = time.perf_counter()
        for _ in range(int(duration_sec / 0.1)):
            self.pub_tip_vel.publish(send_linear_velocity[0], send_linear_velocity[1], send_linear_velocity[2])
            rospy.sleep(0.1)
        end = time.perf_counter()
        self.pub_tip_vel.publish(0, 0, 0)
        print(f'took {end-start}')

    def stop(self):
        for i in range(10):
            self.pub_tip_vel.publish(0,0,0)
            rospy.sleep(0.1)


# if __name__ == '__main__':
#     controller = RobotController()
#     controller.move_forward_needle_axis(kp_linear_vel=2, linear_vel=0.1, duration_sec=2)
    # controller.move_backward_needle_axis()

    # controller.pub_tip_vel(0,0,0)