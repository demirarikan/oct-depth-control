import cv2
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from skimage.measure import LineModelND, ransac
from skimage.restoration import inpaint


class OctPointCloud:

    def __init__(self, seg_volume):
        self.seg_volume = seg_volume
        self.needle_points, self.ilm_points, self.rpe_points = (
            self.__find_first_occurrences(seg_volume)
        )
        self.cleaned_needle_points = None
        self.ilm_inpaint, self.rpe_inpaint = None, None

    def __find_first_occurrences(self, seg_volume, labels=[1, 2, 3]):
        output_coordinates = []
        for value in labels:
            # Create a mask where the values are equal to the target value
            # all desired values will be set to 1 and all others to 0
            mask = (seg_volume == value).astype(
                np.uint8
            )  # uint8 because its slightly faster
            # set 0 values to 5 to avoid them being the minimum
            mask[mask == 0] = 5

            # Find the first occurrence by searching along the first axis (depth axis)
            # argmin returns first index if multiple minimum values are present
            # idx matrix non zero values will contain the first occurrence of the target value in each a-scan
            # idx has shape (seg_volume.shape[0], seg_volume.shape[2])
            indexes = np.argmin(mask, axis=1)
            rows, cols = np.nonzero(indexes)
            # row and column indexes are the same and the idx[row, col] will give the depth value
            coordinates = [[r, indexes[r, c], c] for r, c in zip(rows, cols)]
            output_coordinates.append(coordinates)

        return output_coordinates[0], output_coordinates[1], output_coordinates[2]
    
    def __needle_detection_scikit_ransac(self):
        np_needle_points = np.asarray(self.needle_points)
        _, inliers = ransac(
            np_needle_points, LineModelND, min_samples=2, residual_threshold=7, max_trials=250
        )
        return np_needle_points[inliers]

    def find_needle_tip(self):
        cleaned_needle_points = self.__needle_detection_scikit_ransac()
        self.cleaned_needle_points = cleaned_needle_points
        deepest_point = np.argmax(cleaned_needle_points[:, 1])
        needle_tip_coords = cleaned_needle_points[deepest_point]
        return needle_tip_coords

    def __get_depth_map(self, seg_index):
        z_dim, _, x_dim = self.seg_volume.shape
        depth_map = np.zeros((z_dim, x_dim))
        if seg_index == 2:
            layer_points = self.ilm_points
        elif seg_index == 3:
            layer_points = self.rpe_points
        else:
            raise ValueError("Invalid segmentation index")
        for point in layer_points:
            depth_map[point[0], point[2]] = point[1]
        return depth_map

    def __inpaint_layer(self, depth_map, debug=False):
        depth_map_max = depth_map.max()
        # normalize
        depth_map = depth_map / depth_map_max
        # create inpainting mask
        inpainting_mask = np.where(depth_map == 0, 1, 0).astype(np.uint8)
        # inpaint
        inpaint_res = cv2.inpaint(
            depth_map.astype(np.float32), inpainting_mask, 3, cv2.INPAINT_NS
        )
        # inpaint_res = inpaint.inpaint_biharmonic(depth_map, inpainting_mask)
        # inpaint_res = set_outliers_to_mean_value(inpaint_res, threshold=0.5)
        if debug:

            def visualize_inpainting(depth_map, mask, inpaint_res):
                import matplotlib.pyplot as plt

                fig, axs = plt.subplots(3)
                axs[0].imshow(depth_map, cmap="gray")
                axs[0].set_title("Original depth map")
                axs[1].imshow(mask, cmap="gray")
                axs[1].set_title("Inpainting mask")
                axs[2].imshow(inpaint_res, cmap="gray")
                axs[2].set_title("Inpainting result")
                plt.show()

            visualize_inpainting(depth_map, inpainting_mask, inpaint_res)
        # denormalize
        inpaint_res = inpaint_res * depth_map_max
        return inpaint_res

    def inpaint_layers(self, debug=False):
        ilm_depth_map = self.__get_depth_map(seg_index=2)
        rpe_depth_map = self.__get_depth_map(seg_index=3)

        inpainted_ilm = self.__inpaint_layer(ilm_depth_map, debug)
        inpainted_rpe = self.__inpaint_layer(rpe_depth_map, debug)

        self.ilm_inpaint = inpainted_ilm
        self.rpe_inpaint = inpainted_rpe

        return inpainted_ilm, inpainted_rpe

    def calculate_needle_tip_depth(
        self, needle_tip_coords, inpainted_ilm, inpainted_rpe
    ):
        needle_tip_depth = needle_tip_coords[1]
        ilm_depth = inpainted_ilm[needle_tip_coords[0], needle_tip_coords[2]]
        rpe_depth = inpainted_rpe[needle_tip_coords[0], needle_tip_coords[2]]
        ilm_rpe_distance = rpe_depth - ilm_depth
        needle_tip_depth_relative = needle_tip_depth - ilm_depth
        needle_tip_depth_relative_percentage = (
            needle_tip_depth_relative / ilm_rpe_distance
        )
        return (
            needle_tip_depth,
            needle_tip_depth_relative,
            needle_tip_depth_relative_percentage,
        )

    # visualization functions
    def __needle_pcd(self, color=[1, 0, 0]):
        needle_pcd = o3d.geometry.PointCloud()
        needle_pcd.points = o3d.utility.Vector3dVector(self.needle_points)
        if color:
            needle_pcd.paint_uniform_color(color)
        return needle_pcd

    def __cleaned_needle(self, color=[1, 0, 0]):
        cleaned_needle_pcd = o3d.geometry.PointCloud()
        cleaned_needle_pcd.points = o3d.utility.Vector3dVector(
            self.cleaned_needle_points
        )
        if color:
            cleaned_needle_pcd.paint_uniform_color(color)
        return cleaned_needle_pcd

    def __ilm_pcd(self, color=[0, 1, 0]):
        ilm_points = []
        for index_x in range(self.ilm_inpaint.shape[0]):
            for index_y in range(self.ilm_inpaint.shape[1]):
                ilm_points.append([index_x, self.ilm_inpaint[index_x, index_y], index_y])

        ilm_pcd = o3d.geometry.PointCloud()
        ilm_pcd.points = o3d.utility.Vector3dVector(ilm_points)
        if color:
            ilm_pcd.paint_uniform_color(color)
        return ilm_pcd

    def __rpe_pcd(self, color=[0, 0, 1]):
        rpe_points = []
        for index_x in range(self.rpe_inpaint.shape[0]):
            for index_y in range(self.rpe_inpaint.shape[1]):
                rpe_points.append([index_x, self.rpe_inpaint[index_x, index_y], index_y])

        rpe_pcd = o3d.geometry.PointCloud()
        rpe_pcd.points = o3d.utility.Vector3dVector(rpe_points)
        if color:
            rpe_pcd.paint_uniform_color(color)
        return rpe_pcd

    # visualization utilities
    def __create_mesh_sphere(self, center, radius=3, color=[1.0, 0.0, 1.0]):
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
            [
                [1.0, 0.0, 0.0, center[0]],
                [0.0, 1.0, 0.0, center[1]],
                [0.0, 0.0, 1.0, center[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mesh_sphere.transform(your_transform)
        return mesh_sphere

    def __create_mesh_cylinder(self, needle_tip_coords, radius=0.3, height=500):
        ascan_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height
        )
        transform = np.array(
            [
                [1, 0, 0, needle_tip_coords[0]],
                [0, 0, 1, needle_tip_coords[1]],
                [0, -1, 0, needle_tip_coords[2]],
                [0, 0, 0, 1],
            ]
        )
        ascan_cylinder.transform(transform)
        return ascan_cylinder

    def create_save_point_cloud(
        self,
        inpainted_ilm,
        inpainted_rpe,
        needle_tip_coords,
        show_cleaned_needle=True,
        show_pcd=False,
        save_path="debug_log",
        save_name="point_cloud",
    ):

        if show_cleaned_needle:
            needle_pcd = self.__cleaned_needle()
        else:
            needle_pcd = self.__needle_pcd()

        ilm_pcd = self.__ilm_pcd(inpainted_ilm)
        rpe_pcd = self.__rpe_pcd(inpainted_rpe)

        needle_tip_sphere = self.__create_mesh_sphere(
            needle_tip_coords, radius=3, color=[1.0, 0.0, 1.0]
        )
        ascan_cylinder = self.__create_mesh_cylinder(
            needle_tip_coords, radius=0.3, height=500
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geo in [
            needle_pcd,
            ilm_pcd,
            rpe_pcd,
            needle_tip_sphere,
            ascan_cylinder,
        ]:
            vis.add_geometry(geo)

        ctr = vis.get_view_control()

        ctr.set_lookat(needle_tip_coords)
        ctr.set_up([0, -1, 0])
        ctr.set_front([1, 0, 0])
        ctr.set_zoom(0.2)

        vis.update_renderer()
        vis.capture_screen_image(f"{save_path}/{save_name}.png", True)
        if show_pcd:
            vis.run()
            vis.destroy_window()

    def draw_geometries(geos):
        o3d.visualization.draw_geometries(geos)
