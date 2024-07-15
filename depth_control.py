import time
import oct_point_cloud
from logger import Logger
from mock_leica import MockLeica
from needle_seg_model import NeedleSegModel

MOCK_MODE = True

if not MOCK_MODE:
    import rospy

    from leica_engine import LeicaEngine
    from robot_controller import RobotController


TARGET_DEPTH_RELATIVE = 0.5
CURRENT_DEPTH_RELATIVE = 0.0
ERROR_RANGE = 0.05

N_BSCANS = 5
DIMS = (0.1, 4)

SAVE_PCD = True


def depth_control_loop(
    target_depth_relative=TARGET_DEPTH_RELATIVE,
    current_depth_relative=CURRENT_DEPTH_RELATIVE,
    error_range=ERROR_RANGE,
    n_bscans=N_BSCANS,
    dims=DIMS,
    save_pcd=SAVE_PCD,
):
    seg_model = NeedleSegModel("weights/best_150_val_loss_0.4428_in_retina.pth")

    if MOCK_MODE:
        leica_reader = MockLeica("/home/demir/Desktop/jhu_project/oct_scans/jun18/2.3")
    else:
        rospy.init_node("depth_controller", anonymous=True)
        robot_controller = RobotController()
        leica_reader = LeicaEngine(
            ip_address="192.168.1.75",
            n_bscans=n_bscans,
            xd=dims[0],
            yd=dims[1],
            zd=3.379,
        )
    logger = Logger()

    while current_depth_relative < target_depth_relative:
        raw_oct_volume, _ = leica_reader.__get_b_scans_volume__()

        logger.log_volume(raw_oct_volume)

        oct_volume = seg_model.preprocess_volume(raw_oct_volume, save_train_img=False)
        seg_volume = seg_model.segment_volume(oct_volume, debug=False)
        seg_volume = seg_model.postprocess_volume(seg_volume)

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
                robot_controller.stop()
                break
            else:
                print("Current depth smaller than target, moving robot")
                robot_controller.move_forward_needle_axis(duration_sec=0.5)
                robot_controller.stop()

    print(
        f"Current calculated depth: {current_depth_relative}, Target depth: {target_depth_relative}"
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        depth_control_loop()
        print(f"Elapsed time: {time.perf_counter() - start_time}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
