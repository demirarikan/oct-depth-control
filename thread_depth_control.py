import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

import mock_components
from depth_calculator import DepthCalculator
from logger import Logger
from needle_seg_model import NeedleSegModel


def get_b_scan_and_update(leica_reader):
    while True:
        latest_scan = leica_reader.fast_get_b_scan_volume()
        with condition:
            if scan_queue.full():
                scan_queue.get()  # Remove the old scan if the queue is full
            scan_queue.put(latest_scan)
            condition.notify()


def process_latest_scan(
    seg_model, depth_calculator, robot_controller, logger, target_depth_relative
):

    oct_volumes = {}
    seg_volumes = {}
    depths = {}
    pcd = {}
    count = 0  # image count

    error_range = 0.05
    print("Processing latest scan")
    while True:
        with condition:
            condition.wait()
            scan = scan_queue.get()

        start_time = time.perf_counter()
        oct_volume = seg_model.preprocess_volume(scan)
        oct_volumes[count] = oct_volume
        seg_volume = seg_model.segment_volume(oct_volume)
        # segmentation result is converted to type uint8 for faster processing in the next steps!!!
        seg_volume = seg_model.postprocess_volume(seg_volume)
        seg_volumes[count] = seg_volume

        current_depth_relative, geo_components = depth_calculator.calculate_depth(
            seg_volume, log_final_pcd=True
        )

        print(f'Current relative depth: {current_depth_relative}.')
        print(f'duration: {time.perf_counter() - start_time}')

        depths[count] = current_depth_relative
        pcd[count] = geo_components

        robot_controller.adjust_movement(current_depth_relative, target_depth_relative)
        
        count += 1

        if (
            (current_depth_relative >= 0
            and abs(current_depth_relative - target_depth_relative) < error_range)
            or current_depth_relative > target_depth_relative
        ):
            robot_controller.stop()
            print(f"Stopping robot at depth {current_depth_relative}")
            logger.save_logs(oct_volumes, seg_volumes, pcd, depths)
            break


def depth_control_loop(target_depth_relative, n_bscans, dims, mock_mode):
    if mock_mode:
        leica_reader = mock_components.LeicaEngineMock(
            "/home/demir/Desktop/jhu_project/oct_scans/jun18/2.3"
        )
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
    depth_calculator = DepthCalculator(None)
    logger = Logger()

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(get_b_scan_and_update, leica_reader)
        executor.submit(
            process_latest_scan,
            seg_model,
            depth_calculator,
            robot_controller,
            logger,
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

    depth_control_loop(
        target_depth_relative=0.3, n_bscans=5, dims=(0.1, 4), mock_mode=mock_mode
    )
