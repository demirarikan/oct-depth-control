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

def create_point_cloud_from_vol(seg_volume, seg_index):
    first_occ_coords, color_vals = get_points_and_colors(seg_volume, values=seg_index)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(first_occ_coords)
    point_cloud.colors = o3d.utility.Vector3dVector(color_vals)
    return point_cloud

def calculate_needle_tip_depth(needle_tip_coords, ilm_coords, rpe_coords):
    needle_tip_depth = needle_tip_coords[1]
    ilm_depth = ilm_coords[1]
    rpe_depth = rpe_coords[1]
    ilm_rpe_distance = rpe_depth - ilm_depth
    needle_tip_depth_relative = needle_tip_depth - ilm_depth
    needle_tip_depth_relative_percentage = needle_tip_depth_relative / ilm_rpe_distance
    return needle_tip_depth, needle_tip_depth_relative, needle_tip_depth_relative_percentage

def find_lowest_point(point_cloud):
    np_points = np.asarray(point_cloud.points)
    lowest_index = np.argmax(np_points, axis=0)[2]
    lowest_coords = np_points[lowest_index, :]
    return lowest_coords


# visualization utilities
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

def draw_geometries(geos):
    o3d.visualization.draw_geometries(geos)


# pyRANSAC3d needle tip finding
# create needle point cloud from segmentation results
# apply RANSAC line to find needle tip
# remove outliers
import pyransac3d as pyrsc

def needle_cloud_line_ransac(needle_point_cloud):
    line_ransac = pyrsc.Line()
    point_cloud_points = np.asarray(needle_point_cloud.points)
    line_a, line_b, inlier_indexes = line_ransac.fit(point_cloud_points, thresh=5, maxIteration=1000)
    line_params = (line_a, line_b)
    return line_params, inlier_indexes

def find_needle_tip_3d_ransac(needle_point_cloud):
    line_params, inlier_indexes = needle_cloud_line_ransac(needle_point_cloud)
    cleaned_needle = needle_point_cloud.select_by_index(inlier_indexes)
    needle_tip_coords = find_lowest_point(cleaned_needle)
    return needle_tip_coords, cleaned_needle


# initial noise removal + largest connected component for needle idea
def remove_outliers(point_cloud, nb_points=5, radius=4):
    cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return point_cloud.select_by_index(ind)

def get_largest_cluster(point_cloud, eps=5, min_points=10):
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    largest_cluster_label =  Counter(labels).most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)
    return point_cloud.select_by_index(largest_cluster_indices[0])

def needle_cloud_find_needle_tip(needle_point_cloud, return_clean_point_cloud=False):
    needle_point_cloud = remove_outliers(needle_point_cloud, nb_points=5, radius=10)
    needle_point_cloud = get_largest_cluster(needle_point_cloud, eps=5, min_points=10)
    needle_tip_coords = find_lowest_point(needle_point_cloud)
    if return_clean_point_cloud:
        return needle_tip_coords, needle_point_cloud
    else:
        return needle_tip_coords


# needle point cloud finding based on oriented bounding box angles relative to needle
# this was not as good as RANSAC registration below
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
    for geo in [oct_point_cloud, cleaned_needle_point_cloud, needle_tip_sphere, ascan_cylinder]: # 
        vis.add_geometry(geo)

    ctr = vis.get_view_control()

    ctr.set_lookat(needle_tip_coords)
    ctr.set_up([0, -1, 0])
    ctr.set_front([1, 0, 0])
    ctr.set_zoom(0.2)

    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    vis.capture_screen_image(f'{save_path}/{save_name}.png', True)

def cluster_and_visualize_with_oriented_bboxes(pcd, eps=5, min_points=10):
    # Perform Euclidean clustering
    cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    # Extract clusters
    max_label = cluster_labels.max()
    clusters = []
    for label in range(max_label + 1):
        cluster_points = np.asarray(pcd.points)[cluster_labels == label]
        clusters.append(cluster_points)

    # Visualize clusters with oriented bounding boxes
    vis_list = [pcd]  # List to hold geometries for visualization

    for cluster in clusters:
        # Create a PointCloud object for the cluster
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

        # Compute oriented bounding box
        obb = cluster_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 1)

        x, y, z = compute_tilt_angles_from_obb(obb)
        # only show obbs that have similar tilt to needle
        print((x, y, z))
        if abs(y) - 38 < 8:
            vis_list.append(obb)
        # Visualize the cluster
        vis_list.append(cluster_pcd)

    # Visualization
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    vis_list.append(coords)
    o3d.visualization.draw_geometries(vis_list)

def find_clusters_with_needle_angle(needle_point_cloud, needle_angle, angle_tolerance=10, eps=5, min_points=10):
    cluster_labels = np.array(needle_point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = cluster_labels.max()
    clusters = []
    for label in range(max_label + 1):
        cluster_points = np.asarray(needle_point_cloud.points)[cluster_labels == label]
        clusters.append(cluster_points)

    needle_cluster_points = np.empty((0, 3))
    for cluster in clusters:
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

        try:
            obb = cluster_pcd.get_oriented_bounding_box()
        except RuntimeError:
            continue
        _, y_angle, _ = compute_tilt_angles_from_obb(obb)

        if abs(y_angle) - needle_angle < angle_tolerance:
            needle_cluster_points = np.vstack((needle_cluster_points, cluster))

    return needle_cluster_points

def compute_tilt_angles_from_obb(obb):
    rotation_matrix = np.array(obb.R)

    # Compute tilt angles (euler angles)
    # Open3D uses rotation around Z-Y-X axis convention (intrinsics)
    theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    theta_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles to degrees for easier interpretation
    theta_x_deg = np.degrees(theta_x)
    theta_y_deg = np.degrees(theta_y)
    theta_z_deg = np.degrees(theta_z)

    return theta_x_deg, theta_y_deg, theta_z_deg

def best_fit_line(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    direction_vector = eigenvectors[:, np.argmax(eigenvalues)]
    
    return centroid, direction_vector


# open3d RANSAC with cylindrical needle shape estimate
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def create_cylinder_pcd(radius=4, height=250, number_of_points=400, euler_angles=np.array([0, 35, 0])):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    rotation_radians = np.radians(euler_angles)
    rotation_matrix = cylinder.get_rotation_matrix_from_axis_angle(rotation_radians)
    cylinder.rotate(rotation_matrix, center=(0, 0, 0))
    pcd = cylinder.sample_points_uniformly(number_of_points=number_of_points)
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0,1,0] for _ in range(number_of_points)]))
    return pcd

def register_using_ransac(oct_needle_pcd, cylinder_pcd, voxel_size=0.4):
    target_down, target_fpfh = preprocess_point_cloud(oct_needle_pcd, voxel_size)
    source_down, source_fpfh = preprocess_point_cloud(cylinder_pcd, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    cylinder_pcd.transform(result_ransac.transformation)
    return cylinder_pcd

def create_needle_estimate_pcd(centroid, direction_vector, length=2000, radius=0.5):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction_vector)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction_vector)
    
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    rotation_matrix_3x3 = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation_matrix_3x3
    
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = centroid
    
    transformation_matrix = translation_matrix @ rotation_matrix
    cylinder.transform(transformation_matrix)
    
    return cylinder.sample_points_uniformly(number_of_points=1000)

def outlier_detection_needle_estimate(needle_point_cloud, needle_estimate_point_cloud, radius=5):
    distances = needle_point_cloud.compute_point_cloud_distance(needle_estimate_point_cloud)
    outliers = np.asarray(distances) > radius
    return needle_point_cloud.select_by_index(np.where(outliers == 1)[0], invert=True)


# Layer inpainting
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

def inpaint_layers(ilm_depth_map, rpe_depth_map, debug=False):
    ilm_depth_map_max = ilm_depth_map.max()
    rpe_depth_map_max = rpe_depth_map.max()
    # normalize
    ilm_depth_map = (ilm_depth_map / ilm_depth_map_max)
    rpe_depth_map = (rpe_depth_map / rpe_depth_map_max)
    # create inpainting masks
    ilm_inpainting_mask = np.where(ilm_depth_map == 0, 1, 0).astype(np.uint8)
    rpe_inpainting_mask = np.where(rpe_depth_map == 0, 1, 0).astype(np.uint8)
    # inpaint
    inpaint_ilm = cv2.inpaint(ilm_depth_map.astype(np.float32), ilm_inpainting_mask, 3, cv2.INPAINT_NS)
    inpaint_rpe = cv2.inpaint(rpe_depth_map.astype(np.float32), rpe_inpainting_mask, 3, cv2.INPAINT_NS)
    # inpaint_ilm = inpaint.inpaint_biharmonic(ilm_depth_map, ilm_inpainting_mask)
    # inpaint_rpe = inpaint.inpaint_biharmonic(rpe_depth_map, rpe_inpainting_mask)    
    # inpaint_ilm = set_outliers_to_mean_value(inpaint_ilm, threshold=0.5)
    # inpaint_rpe = set_outliers_to_mean_value(inpaint_rpe, threshold=0.6)

    if debug:
        visualize_inpainting(ilm_depth_map, ilm_inpainting_mask, inpaint_ilm)
        visualize_inpainting(rpe_depth_map, rpe_inpainting_mask, inpaint_rpe)

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

def set_outliers_to_mean_value(array, threshold=0.4, group_size=4):
    rows, cols = array.shape
    if cols % group_size != 0:
        raise ValueError("The number of columns is not divisible by the group size")

    num_groups = cols // group_size

    for i in range(num_groups):
        group = array[:, i*group_size:(i+1)*group_size]

        valid_values = group[group > threshold]
        if valid_values.size > 0:
            average = np.mean(valid_values)
        else:
            average = 1

        group[group <= threshold] = average

    return array

def visualize_inpainting(depth_map, mask, inpaint_res):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3)
    axs[0].imshow(depth_map, cmap='gray')
    axs[0].set_title('Original depth map')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Inpainting mask')
    axs[2].imshow(inpaint_res, cmap='gray')
    axs[2].set_title('Inpainting result')
    plt.show()

def poisson_reconstruction(pcd, depth=9):
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return poisson_mesh

def inpaint_layer(volume):
    layer_pcd = create_point_cloud_from_vol(volume, seg_index=[2])
    layer_pcd.estimate_normals()
    radii = [0.005, 0.1, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(layer_pcd, 
                                                                               o3d.utility.DoubleVector(radii))


    layer_pcd.transform(np.array([[1, 0, 0, 100],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]))
    draw_geometries([rec_mesh, layer_pcd])