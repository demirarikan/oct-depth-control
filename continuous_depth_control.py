import time
import oct_point_cloud
from logger import Logger
from mock_leica import MockLeica
from needle_seg_model import NeedleSegModel

MOCK_MODE = False

if not MOCK_MODE:
    import rospy

    from leica_engine import LeicaEngine
    from robot_controller import RobotController


def depth_control_loop(
    robot_controller,
    seg_model,
    leica_reader,
    logger,
    target_depth_relative=0.5,
    error_range=0.05,
    save_pcd=True,
):  
    # logging flags
    log_raw_oct = True
    log_seg_res = True
    log_final_res = True

    # robot_controller.start_cont_insertion()
    current_depth_relative = 0.0
    try:
        while current_depth_relative < target_depth_relative:
            raw_oct_volume, _ = leica_reader.__get_b_scans_volume__()

            if log_raw_oct:
                logger.log_volume(raw_oct_volume)

            oct_volume = seg_model.preprocess_volume(raw_oct_volume)
            seg_volume = seg_model.segment_volume(oct_volume)
            seg_volume = seg_model.postprocess_volume(seg_volume)

            if log_seg_res:
                logger.log_seg_results(oct_volume, seg_volume)

            # needle processing
            needle_point_cloud = oct_point_cloud.create_point_cloud_from_vol(
                seg_volume, seg_index=[1]
            )
            _, ransac_inliers = oct_point_cloud.needle_cloud_line_ransac(needle_point_cloud)
            cleaned_needle = needle_point_cloud.select_by_index(ransac_inliers)
            needle_tip_coords = oct_point_cloud.find_lowest_point(cleaned_needle)

            # ILM RPE layers processing
            ilm_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=2)
            rpe_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=3)

            ilm_points, rpe_points = oct_point_cloud.inpaint_layers(
                ilm_depth_map, rpe_depth_map, debug=False
            )

            ilm_tip_coords = ilm_points[
                (ilm_points[:, 0] == needle_tip_coords[0])
                & (ilm_points[:, 2] == needle_tip_coords[2])
            ]
            rpe_tip_coords = rpe_points[
                (rpe_points[:, 0] == needle_tip_coords[0])
                & (rpe_points[:, 2] == needle_tip_coords[2])
            ]

            _, _, current_depth_relative = oct_point_cloud.calculate_needle_tip_depth(
                needle_tip_coords, ilm_tip_coords[0], rpe_tip_coords[0]
            )

            if log_final_res:
                logger.log_result_oct(needle_tip_coords, current_depth_relative)

            print(f"Current depth: {current_depth_relative}")

            if save_pcd:
                oct_point_cloud.create_save_point_cloud(
                    cleaned_needle,
                    ilm_points,
                    rpe_points,
                    needle_tip_coords,
                    show_pcd=False,
                    save_path=logger.get_pcd_save_dir(),
                    save_name=logger.get_pcd_save_name(),
                )

            logger.increment_image_count()

            if not MOCK_MODE:
                if (
                    current_depth_relative >= 0
                    and abs(current_depth_relative - target_depth_relative) < error_range
                ):
                    robot_controller.stop_cont_insertion()
                    # robot_controller.stop()
                    break

        robot_controller.stop_cont_insertion()
        print(
            f"Current calculated depth: {current_depth_relative}, Target depth: {target_depth_relative}"
        )
    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == "__main__":
    # Constants
    TARGET_DEPTH_RELATIVE = 0.5
    CURRENT_DEPTH_RELATIVE = 0.0
    ERROR_RANGE = 0.05
    SAVE_PCD = True
    # Scan information
    N_BSCANS = 5
    DIMS = (0.1, 4)

    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")
    from datetime import datetime
    logger = Logger(run_name= "cont_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    if MOCK_MODE:
        leica_reader = MockLeica("/home/demir/Desktop/jhu_project/oct_scans/jun18/2.3")
    else:
        robot_controller = RobotController()
        rospy.init_node("depth_controller", anonymous=True)
        leica_reader = LeicaEngine(
            ip_address="192.168.1.75",
            n_bscans=N_BSCANS,
            xd=DIMS[0],
            yd=DIMS[1],
            zd=3.379,
        )

    def run_loop():
        depth_control_loop(
            robot_controller=robot_controller,
            seg_model=seg_model,
            leica_reader=leica_reader,
            logger=logger,
            target_depth_relative=TARGET_DEPTH_RELATIVE,
            error_range=ERROR_RANGE,
            save_pcd=SAVE_PCD,
    )

    start_time = time.perf_counter()
    try:        
        # depth_control_loop(
        #     robot_controller=robot_controller,
        #     seg_model=seg_model,
        #     leica_reader=leica_reader,
        #     logger=logger,
        #     target_depth_relative=TARGET_DEPTH_RELATIVE,
        #     error_range=ERROR_RANGE,
        #     save_pcd=SAVE_PCD,
        # )
        import cProfile
        cProfile.runctx('run_loop()', globals(), locals(), 'profile_output.prof')
        print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")