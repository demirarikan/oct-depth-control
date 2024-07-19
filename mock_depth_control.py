import time
from datetime import datetime

from depth_controller import DepthController
from logger import Logger
from mock_leica import MockLeica
from needle_seg_model import NeedleSegModel

if __name__ == "__main__":
    # Constants
    TARGET_DEPTH_RELATIVE = 0.5
    ERROR_RANGE = 0.05
    # Scan information
    N_BSCANS = 5
    DIMS = (0.1, 4)

    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")
    logger = Logger(run_name="mock_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    leica_reader = MockLeica(
        "/home/demir/Desktop/jhu_project/oct_scans/debug_log_jul16/cont_20240716-124359/oct_volumes"
    )

    depth_controller = DepthController(
        leica_reader=leica_reader,
        robot_controller=None,
        seg_model=seg_model,
        logger=logger,
    )

    start_time = time.perf_counter()
    try:
        # depth_controller.start_cont_insertion()
        while True:
            current_depth_relative = depth_controller.calculate_depth(
                log_raw_oct=False, log_seg_res=False, log_final_res=False, save_pcd=False
            )
            if (
                current_depth_relative >= 0
                and abs(current_depth_relative - TARGET_DEPTH_RELATIVE) < ERROR_RANGE
            ):
                # robot_controller.stop()
                print(f"Stopping robot at depth {current_depth_relative}")
                break
        print(f"Took {time.perf_counter() - start_time:.2f} seconds")
        logger.save_logs()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        logger.save_logs()
    except StopIteration:
        print("Stopping! Mock Leica out of scans")
        logger.save_logs()
