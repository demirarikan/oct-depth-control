import leica_engine
import oct_point_cloud
import torch
import monai

target_depth = 0.5
current_depth = 0.0

n_bscans = 5
dims = (0.1, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    kernel_size=3,
).to(device)

model.load_state_dict(torch.load('weights/straight_needle_seg_model.pth'))

if __name__ == '__main__':

    leica_engine = leica_engine.LeicaEngine(ip_address="192.168.1.75",   
                               n_bscans=n_bscans, 
                               xd=dims[0], 
                               yd=dims[1], 
                               zd=3.379,
                               )
    
    while current_depth < target_depth:
        oct_volume = leica_engine.__get_b_scans_volume__()

        oct_volume = torch.tensor(oct_volume).to(device)
        with torch.no_grad():
            seg_volume = model(oct_volume)
            seg_volume = torch.argmax(torch.softmax(seg_volume, dim=1), dim=1)
            seg_volume = seg_volume.cpu().numpy()

        needle_point_cloud = oct_point_cloud.create_point_cloud_from_vol(seg_volume, seg_index=[1])
        needle_tip_coords, _ = oct_point_cloud.needle_cloud_find_needle_tip(needle_point_cloud, 
                                                                            return_clean_point_cloud=False)
        
        ilm_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=2)
        rpe_depth_map = oct_point_cloud.get_depth_map(seg_volume, seg_index=3)

        ilm_points, rpe_points = oct_point_cloud.inpaint_layers(ilm_depth_map, rpe_depth_map)

        ilm_tip_coords = ilm_points[(ilm_points[:, 0] == needle_tip_coords[0]) & 
                                    (ilm_points[:, 2] == needle_tip_coords[2])]
        
        rpe_tip_coords = rpe_points[(rpe_points[:, 0] == needle_tip_coords[0]) & 
                                    (rpe_points[:, 2] == needle_tip_coords[2])]
        
        _, _, current_depth = oct_point_cloud.calculate_needle_tip_depth(needle_tip_coords, 
                                                                        ilm_tip_coords[0], 
                                                                        rpe_tip_coords[0])
        print(f"Current depth: {current_depth}")
        
    print("Reached target depth")


