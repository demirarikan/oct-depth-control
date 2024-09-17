import os
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d


class Logger:
    def __init__(
        self,
        log_dir="debug_log",
        run_name=datetime.now().strftime("%Y%m%d-%H%M%S"),
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.__run_dir = os.path.join(log_dir, run_name)
        self.__create_save_dirs()
        self.__image_count = 0

    def __create_save_dirs(self):
        self.oct_volumes_dir = os.path.join(self.__run_dir, "oct_volumes")
        self.seg_images_dir = os.path.join(self.__run_dir, "seg_images")
        self.result_oct_dir = os.path.join(self.__run_dir, "result_oct")
        self.result_pcd_dir = os.path.join(self.__run_dir, "result_pcd")

    def increment_image_count(self):
        self.__image_count += 1

    def log_volume(self, volume, image_count=None):
        os.makedirs(self.oct_volumes_dir, exist_ok=True)
        if image_count is None:
            image_count = self.image_count

        if volume.ndim == 4:
            volume = volume.cpu().numpy().squeeze()

        np.save(
            os.path.join(self.oct_volumes_dir, f"volume_{image_count}.npy"),
            volume,
        )

    def log_seg_results(self, oct_volume, seg_volume, image_count=None):
        os.makedirs(self.seg_images_dir, exist_ok=True)
        if image_count is None:
            image_count = self.image_count

        if oct_volume.ndim == 4:
            oct_volume = oct_volume.cpu().numpy().squeeze()[:, :, 12:-12] * 255

        for idx, seg_mask in enumerate(seg_volume):
            oct_img = oct_volume[idx]
            blended_image = self.__overlay_seg_results(oct_img, seg_mask)
            cv2.imwrite(
                os.path.join(self.seg_images_dir, f"vol_{image_count}_img_{idx}.png"),
                blended_image,
            )

    def log_result_oct(
        self,
        oct_volume,
        seg_volume,
        needle_tip_coords,
        current_depth,
        image_count=None,
    ):
        os.makedirs(self.result_oct_dir, exist_ok=True)
        if image_count is None:
            image_count = self.__image_count

        if oct_volume.ndim == 4:
            oct_volume = oct_volume.cpu().numpy().squeeze()[:, :, 12:-12] * 255
        
        needle_tip_coords = needle_tip_coords.astype(int)
        needle_tip_image = oct_volume[needle_tip_coords[0]]
        needle_tip_seg = seg_volume[needle_tip_coords[0]]
        blended_image = self.__overlay_seg_results(needle_tip_image, needle_tip_seg)
        cv2.circle(
            blended_image,
            (needle_tip_coords[2], needle_tip_coords[1]),
            5,
            (255, 0, 255),
            -1,
        )
        # cv2.putText(
        #     blended_image,
        #     f"Relative depth: {current_depth:.3f}, Needle tip: {needle_tip_coords[0], needle_tip_coords[2], needle_tip_coords[1]}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 0, 255),
        #     2,
        # )
        cv2.imwrite(
            os.path.join(self.result_oct_dir, f"needle_tip_{image_count}.png"),
            blended_image,
        )

    def log_pcd(self, geometries, needle_tip_coords, image_count=None):
        os.makedirs(self.result_pcd_dir, exist_ok=True)
        if image_count is None:
            image_count = self.__image_count

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geo in geometries:
            vis.add_geometry(geo)

        ctr = vis.get_view_control()

        ctr.set_lookat(needle_tip_coords)
        ctr.set_up([0, -1, 0])
        ctr.set_front([1, 0, 0])
        ctr.set_zoom(0.2)

        vis.update_renderer()
        # vis.run()
        # vis.destroy_window()
        vis.capture_screen_image(f"{self.result_pcd_dir}/pcd_{image_count}.png", True)

    def save_logs(self, oct_volumes_dict, seg_volumes_dict, pcd_dict, depths_dict):
        print('Started saving images')
        coords = []
        for count in oct_volumes_dict.keys():
            oct_volume = oct_volumes_dict[count]
            seg_volume = seg_volumes_dict[count]
            geometries = pcd_dict[count]
            depth = depths_dict[count]
            coordinates, geometries = geometries[0:3], geometries[3:]
            needle_tip_coords = coordinates[2]
            coords.append(np.append(needle_tip_coords, coordinates[0:2]))

            self.log_volume(oct_volume, count)
            self.log_seg_results(oct_volume, seg_volume, count)
            self.log_result_oct(oct_volume, seg_volume, needle_tip_coords, depth, count)
            self.log_pcd(geometries, needle_tip_coords, count)
        col_names = "needle_slice, needle_depth, needle_width, ilm_depth, rpe_depth"
        np.savetxt(os.path.join(self.__run_dir, 'coords.csv'), np.array(coords).astype(int), fmt='%i', delimiter=',', header=col_names)
        print('Done saving images!')

    def __overlay_seg_results(self, oct_img, seg_mask, opacity=0.6):
        oct_img_rgb = cv2.cvtColor(oct_img, cv2.COLOR_GRAY2RGB)
        seg_mask = apply_color_map(seg_mask)
        blended_image = cv2.addWeighted(
            oct_img_rgb, 1 - opacity, seg_mask.astype(np.float32), opacity, 0
        )
        return blended_image


def apply_color_map(seg_mask):
    color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

    # Define colors for each segment value (0: black, 1: red, 2: green, 3: blue)
    colors = {
        0: [0, 0, 0],  # black for background
        1: [255, 0, 0],  # red
        2: [0, 255, 0],  # green
        3: [0, 0, 255],  # blue
    }

    for val, color in colors.items():
        color_image[seg_mask == val] = color

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image
