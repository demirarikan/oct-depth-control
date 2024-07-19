from oct_point_cloud import OctPointCloud

class DepthController():
    def __init__(self, leica_reader, robot_controller, seg_model, logger):
        self.leica_reader = leica_reader
        self.robot_controller = robot_controller
        self.seg_model = seg_model
        self.logger = logger

    def calculate_depth(self, log_raw_oct=False, log_seg_res=False, log_final_res=False, save_pcd=False, verbose=True):
        raw_oct_volume, _ = self.leica_reader.__get_b_scans_volume__()

        if log_raw_oct:
            self.logger.log_volume(raw_oct_volume)

        oct_volume = self.seg_model.preprocess_volume(raw_oct_volume)
        seg_volume = self.seg_model.segment_volume(oct_volume)
        # segmentation result is converted to type uint8 for performance!!!
        seg_volume = self.seg_model.postprocess_volume(seg_volume)

        if log_seg_res:
            self.logger.log_seg_results(oct_volume, seg_volume)

        oct_pcd = OctPointCloud(seg_volume=seg_volume)
        needle_tip_coords = oct_pcd.find_needle_tip()
        inpainted_ilm, inpainted_rpe = oct_pcd.inpaint_layers(debug=False)
        _, _, current_depth_relative = oct_pcd.calculate_needle_tip_depth(
            needle_tip_coords, inpainted_ilm, inpainted_rpe
        )

        if verbose:
            print(f"Current depth relative: {current_depth_relative:.4f}")

        if log_final_res:
            self.logger.log_result_oct(needle_tip_coords, current_depth_relative)

        if save_pcd:
            oct_pcd.create_save_point_cloud(
                inpainted_ilm,
                inpainted_rpe,
                needle_tip_coords,
                show_cleaned_needle=True,
                show_pcd=False,
                save_path=self.logger.get_pcd_save_dir(),
                save_name=self.logger.get_pcd_save_name(),
            )

        self.logger.increment_image_count()
        return current_depth_relative

    def start_cont_insertion(self):
        self.robot_controller.start_cont_insertion()

    def stop_cont_insertion(self):
        self.robot_controller.stop_cont_insertion()

    def increment_robot(self, kp_linear_vel, linear_vel, duration_sec):
        self.robot_controller.move_forward_needle_axis(
            kp_linear_vel=kp_linear_vel, 
            linear_vel=linear_vel, 
            duration_sec=duration_sec
        )

    def stop_robot(self):
        self.robot_controller.stop()

