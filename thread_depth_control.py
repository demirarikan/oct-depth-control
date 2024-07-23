import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import mock_components
from depth_calculator import DepthCalculator
from logger import Logger
from needle_seg_model import NeedleSegModel


def get_b_scan_and_update(leica_reader):
    while True:
        latest_scan, _ = leica_reader.__get_b_scans_volume__()
        with condition:
            if scan_queue.full():
                scan_queue.get()  # Remove the old scan if the queue is full
            scan_queue.put(latest_scan)
            condition.notify()


def process_latest_scan(
    seg_model, depth_controller, robot_controller, target_depth_relative
):
    error_range = 0.05
    print("Processing latest scan")
    while True:
        with condition:
            condition.wait()
            scan = scan_queue.get()

        oct_volume = seg_model.preprocess_volume(scan)
        seg_volume = seg_model.segment_volume(oct_volume)
        # segmentation result is converted to type uint8 for performance!!!
        seg_volume = seg_model.postprocess_volume(seg_volume)

        current_depth_relative = depth_controller.calculate_depth(seg_volume)

        robot_controller.adjust_movement(current_depth_relative, target_depth_relative)

        if (
            current_depth_relative >= 0
            and abs(current_depth_relative - target_depth_relative) < error_range
            or current_depth_relative > target_depth_relative
        ):
            robot_controller.stop()
            print(f"Stopping robot at depth {current_depth_relative}")
            break


def depth_control_loop(target_depth_relative, n_bscans, dims, mock_mode):
    if mock_mode:
        leica_reader = mock_components.LeicaEngineMock()
        robot_controller = mock_components.RobotControllerMock()
    else:
        leica_reader = LeicaEngine(
            ip_address="192.168.1.75",
            n_bscans=n_bscans,
            xd=dims[0],
            yd=dims[1],
            zd=3.379,
        )
        robot_controller = RobotController()

    
    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")
    depth_calculator = DepthCalculator()

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(get_b_scan_and_update, leica_reader)
        executor.submit(
            process_latest_scan,
            seg_model,
            depth_calculator,
            robot_controller,
            target_depth_relative,
        )


if __name__ == "__main__":
    mock_mode = False

    if not mock_mode:
        import rospy
        from leica_engine import LeicaEngine
        from robot_controller import RobotController

        rospy.init_node("depth_controller", anonymous=True)


    condition = threading.Condition()
    scan_queue = queue.Queue(maxsize=1)

    depth_control_loop(target_depth_relative=0.5, n_bscans=5, dims=(0.1, 4), mock_mode=mock_mode)
