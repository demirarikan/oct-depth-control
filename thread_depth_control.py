import time
from datetime import datetime

# import rospy

from depth_controller import DepthController
# from leica_engine import LeicaEngine
from logger import Logger
from needle_seg_model import NeedleSegModel
# from robot_controller import RobotController
from mock_leica import MockLeica


import threading
from concurrent.futures import ThreadPoolExecutor
import queue

if __name__ == "__main__":
    # Constants
    TARGET_DEPTH_RELATIVE = 0.5
    ERROR_RANGE = 0.05
    # Scan information
    N_BSCANS = 5
    DIMS = (0.1, 4)

    # rospy.init_node("depth_controller", anonymous=True)

    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")
    logger = Logger(run_name="cont_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # robot_controller = RobotController()
    # leica_reader = LeicaEngine(
    #     ip_address="192.168.1.75",
    #     n_bscans=N_BSCANS,
    #     xd=DIMS[0],
    #     yd=DIMS[1],
    #     zd=3.379,
    # )
    leica_reader = MockLeica(
        "/home/demir/Desktop/jhu_project/oct_scans/jun11/2.3"
    )

    depth_controller = DepthController(
        robot_controller=None,
        seg_model=seg_model,
        logger=logger,
    )

    scan_queue = queue.Queue(maxsize=1)
    
    def get_b_scan_and_update():
        print('Getting B scan and updating')
        while True:
            latest_scan, _ = leica_reader.__get_b_scans_volume__()
            scan_queue.put(latest_scan)
            scan_event.set()

    def process_latest_scan():
        print('Processing latest scan')
        while True:
            scan_event.wait()
            scan = scan_queue.get()
            current_depth_relative = depth_controller.calculate_depth(
                scan, log_raw_oct=False, log_seg_res=False, log_final_res=False, save_pcd=False
            )
            scan_event.clear()
            if (
                current_depth_relative >= 0
                and abs(current_depth_relative - TARGET_DEPTH_RELATIVE) < ERROR_RANGE
                or current_depth_relative > TARGET_DEPTH_RELATIVE
            ):
                # robot_controller.stop()
                print(f'Stopping robot at depth {current_depth_relative}')
                break

    def start():
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(get_b_scan_and_update)
            executor.submit(process_latest_scan)
    
    scan_event = threading.Event()
    start()