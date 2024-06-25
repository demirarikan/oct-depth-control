from collections import Counter

import cv2
import numpy as np
import open3d as o3d
from skimage.restoration import inpaint


def get_points_and_colors(volume, values=[1, 2, 3]):
    z_dim, x_dim, _ = volume.shape
    first_occurrences = np.empty((0,3))
    point_colors = np.empty((0,3))
    for z in range(z_dim):
        for x in range(x_dim):
            ascan = volume[z, :, x]
            for seg_id in values:
                if seg_id == 1:
                    color = np.array([1, 0, 0])
                elif seg_id == 2:
                    color = np.array([0, 1, 0])
                elif seg_id == 3:
                    color = np.array([0, 0, 1])
                else:
                    color = np.array([0, 0, 0])
                
                first_occurrence = np.argwhere(ascan==seg_id)
                if first_occurrence.size > 0:
                    first_occurrences = np.vstack((first_occurrences, np.array([z, first_occurrence[0][0], x])))
                    point_colors = np.vstack((point_colors, color))

    return first_occurrences, point_colors

def get_depth_map(volume, seg_index):
    z_dim, x_dim, _ = volume.shape
    depth_map = np.zeros((z_dim, x_dim))
    for z in range(z_dim):
        for x in range(x_dim):
            ascan = volume[z, :, x]
            first_occurrence = np.argwhere(ascan==seg_index)
            if first_occurrence.size > 0:
                depth_map[z, x] = first_occurrence[0][0]
    return depth_map

def inpaint_layers(ilm_depth_map, rpe_depth_map):
    ilm_depth_map_max = ilm_depth_map.max()
    rpe_depth_map_max = rpe_depth_map.max()
    # normalize
    ilm_depth_map = (ilm_depth_map / ilm_depth_map_max)
    rpe_depth_map = (rpe_depth_map / rpe_depth_map_max)
    # create inpainting masks
    ilm_inpainting_mask = np.where(ilm_depth_map == 0, 1, 0).astype(np.uint8)
    rpe_inpainting_mask = np.where(rpe_depth_map == 0, 1, 0).astype(np.uint8)
    # inpaint
    # inpaint_ilm = cv2.inpaint(ilm_depth_map.astype(np.float32), ilm_inpainting_mask, 3, cv2.INPAINT_NS)
    # inpaint_rpe = cv2.inpaint(rpe_depth_map.astype(np.float32), rpe_inpainting_mask, 3, cv2.INPAINT_NS)
    inpaint_ilm = inpaint.inpaint_biharmonic(ilm_depth_map, ilm_inpainting_mask)
    inpaint_rpe = inpaint.inpaint_biharmonic(rpe_depth_map, rpe_inpainting_mask)
    # denormalize
    inpaint_ilm = (inpaint_ilm) * ilm_depth_map_max
    inpaint_rpe = (inpaint_rpe) * rpe_depth_map_max

    ilm_points = np.empty((0,3))
    rpe_points = np.empty((0,3))
    for i in range(inpaint_ilm.shape[0]):
        for j in range(inpaint_ilm.shape[1]):
            # ilm and rpe final points for 3d visualization
            ilm_point = np.array([i, inpaint_ilm[i, j], j])
            ilm_points = np.vstack((ilm_points, ilm_point))

            rpe_point = np.array([i, inpaint_rpe[i, j], j])
            rpe_points = np.vstack((rpe_points, rpe_point))

    return ilm_points, rpe_points

def remove_outliers(point_cloud, nb_points=5, radius=4):
    cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return point_cloud.select_by_index(ind)

def get_largest_cluster(point_cloud, eps=5, min_points=10):
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    largest_cluster_label =  Counter(labels).most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)
    return point_cloud.select_by_index(largest_cluster_indices[0])

def find_lowest_point(point_cloud):
    np_points = np.asarray(point_cloud.points)
    lowest_index = np.argmax(np_points, axis=0)[2]
    lowest_coords = np_points[lowest_index, :]
    return lowest_coords

def needle_cloud_find_needle_tip(needle_point_cloud, return_clean_point_cloud=False):
    needle_point_cloud = remove_outliers(needle_point_cloud, nb_points=5, radius=4)
    needle_point_cloud = get_largest_cluster(needle_point_cloud, eps=5, min_points=10)
    needle_tip_coords = find_lowest_point(needle_point_cloud)
    if return_clean_point_cloud:
        return needle_tip_coords, needle_point_cloud
    else:
        return needle_tip_coords

def calculate_needle_tip_depth(needle_tip_coords, ilm_coords, rpe_coords):
    needle_tip_depth = needle_tip_coords[1]
    ilm_depth = ilm_coords[1]
    rpe_depth = rpe_coords[1]
    ilm_rpe_distance = rpe_depth - ilm_depth
    needle_tip_depth_relative = needle_tip_depth - ilm_depth
    needle_tip_depth_relative_percentage = needle_tip_depth_relative / ilm_rpe_distance
    return needle_tip_depth, needle_tip_depth_relative, needle_tip_depth_relative_percentage

def create_point_cloud_from_vol(seg_volume, seg_index):
    needle_first_occ_coords, needle_colors = get_points_and_colors(seg_volume, values=seg_index)
    needle_point_cloud = o3d.geometry.PointCloud()
    needle_point_cloud.points = o3d.utility.Vector3dVector(needle_first_occ_coords)
    # needle_colors = np.array([[1, 0, 0] for _ in range(needle_first_occ_coords.shape[0])])
    needle_point_cloud.colors = o3d.utility.Vector3dVector(needle_colors)
    return needle_point_cloud

def create_mesh_sphere(center, radius=3, color=[1., 0., 1.]):
    """
    Create a mesh sphere with the given center, radius, and color.

    Parameters:
    - center (list): The center coordinates of the sphere in the form [slice, x, y].
    - radius (float): The radius of the sphere.
    - color (list): The color of the sphere in RGB format, with values ranging from 0 to 1.

    Returns:
    - mesh_sphere (o3d.geometry.TriangleMesh): The created mesh sphere.
    """

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.paint_uniform_color(color)

    your_transform = np.asarray(
                    [[1., 0., 0., center[0]],
                    [0., 1., 0.,  center[1]],
                    [0., 0.,  1., center[2]],
                    [0., 0., 0., 1.0]])
    mesh_sphere.transform(your_transform)
    return mesh_sphere 

def create_mesh_cylinder(needle_tip_coords, radius=0.3, height=500):
    ascan_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    transform = np.array([
        [1, 0, 0, needle_tip_coords[0]],
        [0, 0, 1, needle_tip_coords[1]],
        [0, -1, 0, needle_tip_coords[2]],
        [0, 0, 0, 1]
    ])
    ascan_cylinder.transform(transform)
    return ascan_cylinder

def create_save_point_cloud(cleaned_needle_point_cloud, 
                            ilm_points,
                            rpe_points,
                            needle_tip_coords, 
                            save_path='debug_point_cloud_images',
                            save_name='point_cloud'):
    needle_tip_sphere = create_mesh_sphere(needle_tip_coords, radius=3, color=[1., 0., 1.])
    ascan_cylinder = create_mesh_cylinder(needle_tip_coords, radius=0.3, height=500)

    needle_colors = np.array([[1, 0, 0] for _ in range(np.asarray(cleaned_needle_point_cloud.points).shape[0])])
    ilm_colors = np.array([[0, 1, 0] for _ in range(ilm_points.shape[0])])
    rpe_colors = np.array([[0, 0, 1] for _ in range(rpe_points.shape[0])])

    cleaned_needle_point_cloud.colors = o3d.utility.Vector3dVector(needle_colors)

    oct_point_cloud = o3d.geometry.PointCloud()

    oct_point_cloud.points = o3d.utility.Vector3dVector(np.vstack((ilm_points, rpe_points)))
    oct_point_cloud.colors = o3d.utility.Vector3dVector(np.vstack((ilm_colors, rpe_colors)))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in [oct_point_cloud, cleaned_needle_point_cloud, needle_tip_sphere, ascan_cylinder]:
        vis.add_geometry(geo)

    ctr = vis.get_view_control()

    ctr.set_lookat(needle_tip_coords)
    ctr.set_up([0, -1, 0])
    ctr.set_front([1, 0, 0])
    ctr.set_zoom(0.2)

    vis.update_renderer()
    # vis.run()
    # vis.destroy_window()
    vis.capture_screen_image(f'{save_path}/{save_name}.png', True)
