# import leica_engine
from mock_leica import MockLeica
import oct_point_cloud
import torch
from torchvision import transforms
import monai
import numpy as np
from needle_seg_model import NeedleSegModel

target_depth_relative = 0.5
current_depth_relative = 0.0

n_bscans = 5
dims = (0.1, 5)

seg_model = NeedleSegModel('weights/straight_needle_seg_model.pth')

# leica_reader = leica_engine.LeicaEngine(ip_address="192.168.1.75",   
#                                n_bscans=n_bscans, 
#                                xd=dims[0], 
#                                yd=dims[1], 
#                                zd=3.379,
#                                )

leica_reader = MockLeica('/home/demir/Desktop/jhu_project/oct_scans/jun11/2')
print("Leica reader initialized")

if __name__ == '__main__':
    while current_depth_relative < target_depth_relative:
        oct_volume = leica_reader.__get_b_scans_volume__()

        oct_volume = seg_model.prepare_vol_from_leica_engine(oct_volume)

        seg_volume = seg_model.segment_volume(oct_volume)

        needle_point_cloud = oct_point_cloud.create_point_cloud_from_vol(seg_volume, seg_index=[1])
        needle_tip_coords = oct_point_cloud.needle_cloud_find_needle_tip(needle_point_cloud, 
                                                                            return_clean_point_cloud=False)
        
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
        
    print("Reached target depth")


