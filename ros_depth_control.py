import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Bool
from oct_point_cloud import OctPointCloud
from needle_seg_model import NeedleSegModel
import numpy as np
from image_conversion_without_using_ros import image_to_numpy
import time


class ROSDepthControl:
    def __init__(self, target_depth, max_vel, seg_model):
        # ROS components
        self.insertion_vel_pub = rospy.Publisher("cont_mov_vel", Float64, queue_size=1)
        self.insertion_stop_pub = rospy.Publisher("stop_cont_pub", Bool, queue_size=1)
        self.b_scan_sub = rospy.Subscriber("oct_b_scan", Image, self.b_scan_callback, queue_size=3)
        self.cv_bridge = CvBridge()
        self.latest_b5_vol = []

        # insertion parameters
        self.target_depth = target_depth
        self.max_vel = max_vel
        self.insertion_complete = False

        # components
        self.seg_model = seg_model

    def b_scan_callback(self, data):
        b_scan = image_to_numpy(data)
        self.latest_b5_vol.append(b_scan)
        if len(self.latest_b5_vol) == 5 and not self.insertion_complete:
            # start_time = time.perf_counter()
            np_b5_vol = np.array(self.latest_b5_vol)
            seg_vol = self.segment_volume(np_b5_vol)
            needle_tip_coords, inpainted_ilm, inpainted_rpe = self.process_pcd(seg_vol)
            _, needle_depth, _, _ = self.calculate_needle_depth(
                needle_tip_coords, inpainted_ilm, inpainted_rpe
            )
            self.update_insertion_velocity(needle_depth)
            self.latest_b5_vol = []
            # print(f"Took: {time.perf_counter()-start_time} seconds")

    def segment_volume(self, oct_volume):
        oct_volume = self.seg_model.preprocess_volume(oct_volume)
        seg_volume = self.seg_model.segment_volume(oct_volume)
        seg_volume = self.seg_model.postprocess_volume(seg_volume)
        return seg_volume

    def process_pcd(self, seg_volume):
        oct_pcd = OctPointCloud(seg_volume)
        needle_tip_coords = oct_pcd.find_needle_tip()
        inpainted_ilm, inpainted_rpe = oct_pcd.inpaint_layers()
        return needle_tip_coords, inpainted_ilm, inpainted_rpe

    def calculate_needle_depth(self, needle_tip_coords, inpainted_ilm, inpainted_rpe):
        needle_tip_depth = needle_tip_coords[1]
        ilm_depth = inpainted_ilm[needle_tip_coords[0], needle_tip_coords[2]]
        rpe_depth = inpainted_rpe[needle_tip_coords[0], needle_tip_coords[2]]
        ilm_rpe_distance = rpe_depth - ilm_depth
        needle_tip_depth_relative = needle_tip_depth - ilm_depth
        needle_tip_depth_relative_percentage = (
            needle_tip_depth_relative / ilm_rpe_distance
        )
        return (
            needle_tip_depth,
            needle_tip_depth_relative,
            needle_tip_depth_relative_percentage,
            (ilm_depth, rpe_depth),
        )

    def update_insertion_velocity(self, current_depth):
        insertion_vel = self.__calculate_insertion_velocity(current_depth)
        if insertion_vel == 0:
            self.insertion_vel_pub.publish(0)
            self.insertion_stop_pub.publish(True)
            self.insertion_complete = True
        else:
            self.insertion_vel_pub.publish(insertion_vel)


    def __calculate_insertion_velocity(self, current_depth, method="linear"):
        threshold = self.target_depth * 0.1
        difference = abs(self.target_depth - current_depth)
        # Move needle back if it overshoots the target depth
        if current_depth > self.target_depth:
            return -0.2
        # Stop the insertion if the needle is within the threshold
        if difference < threshold:
            return 0
        
        if method == "linear":
            y_intercept = self.max_vel
            x_intercept = self.target_depth
            vel = min(
                self.max_vel, (-(y_intercept / x_intercept) * current_depth) + y_intercept
            )
            vel = max(vel, 0)
        elif method == "exponential":
            vel = min(difference**2, self.max_vel)
        return vel


if __name__ == "__main__":
    rospy.init_node("depth_control")
    target_depth = 0.5
    max_vel = 0.3
    seg_model = NeedleSegModel(None, "weights/best_150_val_loss_0.4428_in_retina.pth")
    depth_control = ROSDepthControl(target_depth, max_vel, seg_model)
    rospy.spin()