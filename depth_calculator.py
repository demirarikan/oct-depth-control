from oct_point_cloud import OctPointCloud


class DepthCalculator:
    def __init__(self, logger):
        self.logger = logger

    def calculate_depth(self, segmented_oct_volume, log_final_pcd=False):
        oct_pcd = OctPointCloud(seg_volume=segmented_oct_volume)
        needle_tip_coords = oct_pcd.find_needle_tip()
        inpainted_ilm, inpainted_rpe = oct_pcd.inpaint_layers(debug=False)
        _, _, current_depth_relative = oct_pcd.calculate_needle_tip_depth(
            needle_tip_coords, inpainted_ilm, inpainted_rpe
        )
        if log_final_pcd:
            # self.logger.log_pcd(
            #     oct_pcd.create_point_cloud_components(needle_tip_coords)
            # )
            components = oct_pcd.create_point_cloud_components(needle_tip_coords)
        return current_depth_relative, components

