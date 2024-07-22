import os
from datetime import datetime

import cv2
import numpy as np


class Logger:
    def __init__(
        self,
        log_dir="debug_log",
        run_name=datetime.now().strftime("%Y%m%d-%H%M%S"),
        save_realtime=False,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.__run_dir = os.path.join(log_dir, run_name)
        self.__create_save_dirs()
        self.__image_count = 0
        self.__save_realtime = save_realtime

        # logs
        self.raw_oct_log = {}
        self.seg_res_log = {}
        self.final_res_log = {}
        self.pcd_log = {}

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
        if self.__save_realtime:
            np.save(
                os.path.join(self.oct_volumes_dir, f"volume_{image_count}.npy"),
                volume,
            )
        else:
            self.raw_oct_log[image_count] = volume

    def log_seg_results(self, oct_volume, seg_volume, image_count=None):
        os.makedirs(self.seg_images_dir, exist_ok=True)
        if image_count is None:
            image_count = self.image_count
        if self.__save_realtime:
            oct_volume = oct_volume.cpu().numpy().squeeze()[:, 12:-12, :] * 255
            for idx, seg_mask in enumerate(seg_volume):
                oct_img = oct_volume[idx]
                blended_image = self.__overlay_seg_results(oct_img, seg_mask)
                cv2.imwrite(
                    os.path.join(
                        self.seg_images_dir, f"vol_{image_count}_img_{idx}.png"
                    ),
                    blended_image,
                )
        else:
            self.seg_res_log[image_count] = (oct_volume, seg_volume)

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

        if self.__save_realtime:
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
            cv2.putText(
                blended_image,
                f"Relative depth: {current_depth:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )
            cv2.imwrite(
                os.path.join(self.result_oct_dir, f"needle_tip_{image_count}.png"),
                blended_image,
            )
        else:
            self.final_res_log[image_count] = (
                oct_volume,
                seg_volume,
                needle_tip_coords,
                current_depth,
            )

    def save_pcd(self, oct_pcd, needle_tip_coords, geometries=None, image_count=None):
        os.makedirs(self.result_pcd_dir, exist_ok=True)
        if image_count is None:
            image_count = self.__image_count

        if geometries is None:
            geometries = oct_pcd.create_point_cloud_components(
                needle_tip_coords, show_cleaned_needle=True
            )

        if self.__save_realtime:
            oct_pcd.save_pcd_visualization(
                geometries,
                needle_tip_coords,
                show_pcd=False,
                save_path=self.result_pcd_dir,
                save_name=f"pcd_{image_count}",
            )
        else:
            self.pcd_log[image_count] = geometries

    def save_logs(self):
        print("Saving logs...")
        if self.raw_oct_log:
            for image_count, raw_oct in self.raw_oct_log.items():
                self.logger.log_volume(raw_oct, image_count=image_count)

        if self.seg_res_log:
            for image_count, (oct_volume, seg_volume) in self.seg_res_log.items():
                self.logger.log_seg_results(
                    oct_volume, seg_volume, image_count=image_count
                )

        if self.final_res_log:
            for image_count, (
                oct_volume,
                seg_volume,
                needle_tip_coords,
                current_depth,
            ) in self.final_res_log:
                self.logger.log_result_oct(
                    oct_volume,
                    seg_volume,
                    needle_tip_coords,
                    current_depth,
                    image_count=image_count,
                )

        if self.pcd_log:
            for image_count, geometries in self.pcd_log.items():
                pass
                # TODO: FIX THIS!!!!!! ADD OCT_PCD OBJECT SOMEWHERE
                # self.logger.save_pcd(
                #     geometries, needle_tip_coords, image_count=image_count
                # )
        print("Logs saved!")

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
