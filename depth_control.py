# import leica_engine
import oct_point_cloud
from mock_leica import MockLeica
from needle_seg_model import NeedleSegModel
# import rospy
# from robot_controller import RobotController

target_depth_relative = 0.5
current_depth_relative = 0.0
error_range = 0.05

n_bscans = 5
dims = (0.1, 5)

save_point_cloud_image = True
image_count = 0

if __name__ == '__main__':
    # rospy.init_node('depth_controller', anonymous=True)

    # robot_controller = RobotController()
    seg_model = NeedleSegModel('weights/straight_needle_seg_model.pth')
    # leica_reader = leica_engine.LeicaEngine(ip_address="192.168.1.75",   
    #                                n_bscans=n_bscans, 
    #                                xd=dims[0], 
    #                                yd=dims[1], 
    #                                zd=3.379,
    #                                )
    leica_reader = MockLeica('/home/demir/Desktop/jhu_project/oct_scans/jun11/2.9')
    print("Leica reader initialized")

    while current_depth_relative < target_depth_relative:
    # while True:
        try:
            oct_volume, _ = leica_reader.__get_b_scans_volume__()

            oct_volume = seg_model.prepare_vol_from_leica_engine(oct_volume, save_train_img=False)

            seg_volume = seg_model.segment_volume(oct_volume, debug=False)

            needle_point_cloud = oct_point_cloud.create_point_cloud_from_vol(seg_volume, seg_index=[1])

            cylinder = oct_point_cloud.create_cylinder_pcd()
            cylinder = oct_point_cloud.register_using_ransac(needle_point_cloud, cylinder, voxel_size=0.8)
            oct_point_cloud.draw_geometries([needle_point_cloud, cylinder])
            cleaned_needle = oct_point_cloud.outlier_detection_needle_estimate(needle_point_cloud, cylinder)
            oct_point_cloud.draw_geometries([cleaned_needle, cylinder])


            needle_tip_coords = oct_point_cloud.find_lowest_point(cleaned_needle)
            
            ilm_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=2)
            rpe_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=3)

            ilm_points, rpe_points = oct_point_cloud.inpaint_layers(ilm_depth_map, rpe_depth_map)

            ilm_tip_coords = ilm_points[(ilm_points[:, 0] == needle_tip_coords[0]) & 
                                        (ilm_points[:, 2] == needle_tip_coords[2])]
            
            rpe_tip_coords = rpe_points[(rpe_points[:, 0] == needle_tip_coords[0]) & 
                                        (rpe_points[:, 2] == needle_tip_coords[2])]
            
            _, _, current_depth_relative = oct_point_cloud.calculate_needle_tip_depth(needle_tip_coords, 
                                                                            ilm_tip_coords[0], 
                                                                            rpe_tip_coords[0])
            print(f"Current depth: {current_depth_relative}")

            if save_point_cloud_image:
                oct_point_cloud.create_save_point_cloud(cleaned_needle,
                                                        ilm_points,
                                                        rpe_points,
                                                        needle_tip_coords,
                                                        save_name=f'needle_tip_{image_count}_{current_depth_relative:.2f}')
                image_count += 1

            if (current_depth_relative >= 0 and 
                current_depth_relative <= target_depth_relative+error_range and 
                current_depth_relative >= target_depth_relative-error_range):
                # robot_controller.stop()
                break
            else:
                print('Current depth smaller than target, moving robot')
                # robot_controller.move_forward_needle_axis(duration_sec=0.5)
                # robot_controller.stop()
        except KeyboardInterrupt:
            break
        
    print(f'Current calculated depth: {current_depth_relative}, Target depth: {target_depth_relative}')


