import cv2
import numpy as np

def find_layers_coords(image, a_scan):
    if a_scan < 0 or a_scan >= image.shape[1]:
        return np.inf, np.inf, np.inf
    try:
        ilm_row = np.where(image[:, a_scan] == 2)[0][0]
        rpe_row = np.where(image[:, a_scan] == 3)[0][0]
        return a_scan, ilm_row, rpe_row
    except IndexError:
        candidate_a_scans = []
        # find all a_scans with both ILM and RPE
        for col_index in range(image.shape[1]):
            column = image[:, col_index]
            if all(value in column for value in [2, 3]):
                candidate_a_scans.append(col_index)
        if not candidate_a_scans:
            return np.inf, np.inf, np.inf
        
        # find closes to input a_scan
        closest_a_scan = min(candidate_a_scans, key=lambda x: abs(x - a_scan))
        ilm_row = np.where(image[:, closest_a_scan] == 2)[0][0]
        rpe_row = np.where(image[:, closest_a_scan] == 3)[0][0]
        return closest_a_scan, ilm_row, rpe_row
    
def calc_normalized_pos_between_layers(ilm_row, rpe_row, needle_tip_row):
    if needle_tip_row < ilm_row:
        return 0
    elif needle_tip_row > rpe_row:
        return 1
    return (needle_tip_row - ilm_row) / (rpe_row - ilm_row)

def volume_find_needle_tip_largest_component(segmented_b_scans, needle_label=1, min_area=150, visualize=False):
    largest_component_area = 0
    needle_tip_largest_component = None
    slice_idx_largest_component = -1
    for slice_idx, b_scan in enumerate(segmented_b_scans):
        # cv2.imwrite(f'bscan_{slice_idx}.png', apply_color_map(b_scan))
        b_scan = cleanup_edges(b_scan)
        # cv2.imwrite(f'cleaned_{slice_idx}.png', apply_color_map(b_scan))

        needle_mask = (b_scan == needle_label).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(needle_mask, connectivity=8)
        # remove components with bounding box area less than min_area
        valid_components = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

        if visualize:
            visualize_component_analysis(b_scan, num_labels, stats, slice_idx=slice_idx)

        if valid_components:
            # tuple with component index and area 
            largest_component = max(valid_components, key=lambda x: x[1])

            if largest_component[1] > largest_component_area:
                component_label, largest_component_area = largest_component
                component_mask = (labels == component_label)
                y_coords, x_coords = np.where(component_mask)
                lowest_pixel = np.argmax(y_coords)
                needle_tip = (x_coords[lowest_pixel], y_coords[lowest_pixel])
                needle_tip_largest_component = needle_tip
                slice_idx_largest_component = slice_idx

    return (slice_idx_largest_component, needle_tip_largest_component)

def cleanup_edges(segmentation_mask):
    eroison_kernel = np.ones((2, 2), np.uint8)
    dilation_kernel = np.ones((3, 3), np.uint8)

    dilated_mask = np.zeros_like(segmentation_mask)

    for label in [2, 3, 1]: # order is important we want to handle needle last
        binary_mask = np.uint8(segmentation_mask == label) 

        eroded_binary_mask = cv2.erode(binary_mask, eroison_kernel, iterations=1) 
        dilated_binary_mask = cv2.dilate(eroded_binary_mask, dilation_kernel, iterations=1) 
        
        dilated_mask[dilated_binary_mask == 1] = label 
        segmentation_mask[dilated_binary_mask == 1] = label

    dilated_mask[segmentation_mask == 0] = 0
    return dilated_mask


def apply_color_map(seg_mask):
    # Create an empty image with 3 channels (RGB)
    color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    
    # Define colors for each segment value (0: black, 1: red, 2: green, 3: blue)
    colors = {
        0: [0, 0, 0],        # black for background
        1: [255, 0, 0],      # red
        2: [0, 255, 0],      # green
        3: [0, 0, 255],      # blue
        # Add more colors if there are more segments
    }
    
    # Apply the colors to the image
    for val, color in colors.items():
        color_image[seg_mask == val] = color

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
def visualize_component_analysis(segmented_b_scan, num_labels, stats, slice_idx=0):
    # custom colormap for segmentation mask 
    colors = ['black', 'red', 'green', 'blue']
    seg_cmap = plt.cm.colors.ListedColormap(colors)

    plt.imshow(segmented_b_scan, cmap=seg_cmap)
    plt.axis('off')

    min_area = 150
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='y', facecolor='none') 
            plt.gca().add_patch(rect)
    plt.savefig(f'component_analysis_{slice_idx}.png')
    plt.show()
    

