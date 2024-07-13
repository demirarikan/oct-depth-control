from datetime import datetime
import os

import numpy as np
import cv2

class Logger():
    def __init__(self, log_dir='debug_log', run_name=datetime.now().strftime('%Y%m%d-%H%M%S')):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.__run_dir = os.path.join(log_dir, run_name)
        self.__create_save_dirs()
        self.__image_count = 0

    def __create_save_dirs(self):
        self.oct_volumes_dir = os.path.join(self.__run_dir, 'oct_volumes')
        self.seg_images_dir = os.path.join(self.__run_dir, 'seg_images')
        self.result_oct_dir = os.path.join(self.__run_dir, 'result_oct')
        self.result_pcd_dir = os.path.join(self.__run_dir, 'result_pcd')
        os.makedirs(self.oct_volumes_dir)
        os.makedirs(self.seg_images_dir)
        os.makedirs(self.result_oct_dir)
        os.makedirs(self.result_pcd_dir)
    
    def increment_image_count(self):
        self.__image_count += 1

    def log_volume(self, volume):
        np.save(os.path.join(self.oct_volumes_dir, f'volume_{self.__image_count}.npy'), volume)
    
    def log_seg_results(self, oct_volume, seg_volume):
        opacity = 0.4
        oct_volume = oct_volume.cpu().numpy().squeeze()[:, 12: -12, :] * 255
        for idx, seg_mask in enumerate(seg_volume):
            oct_img = oct_volume[idx]
            oct_img_rgb = cv2.cvtColor(oct_img, cv2.COLOR_GRAY2RGB)
            seg_mask = apply_color_map(seg_mask)
            blended_image = cv2.addWeighted(oct_img_rgb, 
                                            1-opacity, 
                                            seg_mask.astype(np.float32), 
                                            opacity, 
                                            0)
            cv2.imwrite(
                os.path.join(self.seg_images_dir, f'vol_{self.__image_count}_img_{idx}.png'), 
                blended_image
                )
    
    def log_result_oct(self, needle_tip_coords, current_depth):
        # we already save blended oct+seg images so we can just pick the one with the needle tip
        needle_tip_coords = needle_tip_coords.astype(int)
        needle_tip_image = cv2.imread(
            os.path.join(self.seg_images_dir, f'vol_{self.__image_count}_img_{needle_tip_coords[0]}.png'))
        
        cv2.circle(needle_tip_image, 
                   (needle_tip_coords[2], needle_tip_coords[1]), 
                   5, (255, 0, 255), -1)
        
        cv2.putText(needle_tip_image, 
                    f'Depth: {current_depth:.3f}', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 255), 2)
        
        cv2.imwrite(os.path.join(self.result_oct_dir, f'needle_tip_{self.__image_count}.png'), needle_tip_image)

    def get_pcd_save_dir(self):
        return self.result_pcd_dir
    
    def get_pcd_save_name(self):
        return f'pcd_{self.__image_count}'

def apply_color_map(seg_mask):
    color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    
    # Define colors for each segment value (0: black, 1: red, 2: green, 3: blue)
    colors = {
        0: [0, 0, 0],        # black for background
        1: [255, 0, 0],      # red
        2: [0, 255, 0],      # green
        3: [0, 0, 255],      # blue
    }
    
    for val, color in colors.items():
        color_image[seg_mask == val] = color

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image