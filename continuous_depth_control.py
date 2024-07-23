import time
from datetime import datetime

import rospy

from depth_calculator import DepthCalculator
from leica_engine import LeicaEngine
from logger import Logger
from needle_seg_model import NeedleSegModel
from robot_controller import RobotController

if __name__ == "__main__":
    # Constants
    TARGET_DEPTH_RELATIVE = 0.5
    ERROR_RANGE = 0.05
    # Scan information
    N_BSCANS = 5
    DIMS = (0.1, 4)

    rospy.init_node("depth_controller", anonymous=True)

    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")
    logger = Logger(run_name="cont_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    robot_controller = RobotController()
    leica_reader = LeicaEngine(
        ip_address="192.168.1.75",
        n_bscans=N_BSCANS,
        xd=DIMS[0],
        yd=DIMS[1],
        zd=3.379,
    )

    depth_controller = DepthCalculator(
        leica_reader=leica_reader,
        robot_controller=robot_controller,
        seg_model=seg_model,
        logger=logger,
    )

    start_time = time.perf_counter()
    try:
        depth_controller.start_cont_insertion()
        while True:
            current_depth_relative = depth_controller.calculate_depth(
                log_raw_oct=True, log_seg_res=True, log_final_res=True, save_pcd=True
            )
            if (
                current_depth_relative >= 0
                and abs(current_depth_relative - TARGET_DEPTH_RELATIVE) < ERROR_RANGE
            ):
                robot_controller.stop()
                print(f'Stopping robot at depth {current_depth_relative}')
                break
        print(f'Took {time.perf_counter() - start_time:.2f} seconds')
    except KeyboardInterrupt:
        robot_controller.stop()
        print("KeyboardInterrupt")
